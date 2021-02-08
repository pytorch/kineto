# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import multiprocessing
import os
import threading
import time
from collections import OrderedDict

import werkzeug
from tensorboard.plugins import base_plugin
from werkzeug import wrappers

from . import consts
from . import utils
from .profiler import RunLoader
from .run import Run

logger = utils.get_logger()


class TorchProfilerPlugin(base_plugin.TBPlugin):
    """TensorBoard plugin for Torch Profiler."""

    plugin_name = consts.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates TorchProfilerPlugin.
        Args:
          context: A base_plugin.TBContext instance.
        """
        super(TorchProfilerPlugin, self).__init__(context)
        self.logdir = os.path.abspath(context.logdir)

        self._is_active = None
        self._is_active_initialized_event = threading.Event()

        self._runs = OrderedDict()
        self._runs_lock = threading.Lock()

        self._queue = multiprocessing.Queue()
        monitor_runs = threading.Thread(target=self.monitor_runs, name="monitor_runs", daemon=True)
        monitor_runs.start()

        receive_runs = threading.Thread(target=self.receive_runs, name="receive_runs", daemon=True)
        receive_runs.start()

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.
        """
        self._is_active_initialized_event.wait()
        return self._is_active

    def get_plugin_apps(self):
        return {
            "/index.js": self.static_file_route,
            "/main.js": self.static_file_route,
            "/index.html": self.static_file_route,
            "/overall.html": self.static_file_route,
            "/trace_viewer_full.html": self.static_file_route,
            "/trace_embedding.html": self.static_file_route,
            "/operator.html": self.static_file_route,
            "/kernel.html": self.static_file_route,
            "/runs": self.runs_route,
            "/views": self.views_route,
            "/workers": self.workers_route,
            "/overview": self.overview_route,
            "/operation": self.operation_pie_route,
            "/operation/table": self.operation_table_route,
            "/kernel": self.kernel_pie_route,
            "/kernel/table": self.kernel_table_route,
            "/trace": self.trace_route
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js")

    def monitor_runs(self):
        logger.info("Monitor runs begin")

        # Set _is_active quickly based on file pattern match, don't wait for data loading
        self._is_active = any(self._get_run_dirs())
        self._is_active_initialized_event.set()

        touched = set()
        while True:
            try:
                logger.debug("Scan run dir")
                run_dirs = self._get_run_dirs()

                # Assume no deletion on run directories, trigger async load if find a new run
                for name, run_dir in run_dirs:
                    if name not in touched:
                        logger.info("Find run %s under %s", name, run_dir)
                        touched.add(name)
                        # Use multiprocessing to avoid UI stall and reduce data parsing time
                        process = multiprocessing.Process(target=_load_run, args=(self._queue, name, run_dir))
                        process.daemon = True
                        process.start()
            except Exception as ex:
                logger.warning("Failed to scan runs. Exception=%s", ex, exc_info=True)

            time.sleep(consts.MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS)

    def receive_runs(self):
        while True:
            run = self._queue.get()
            if run is None:
                continue

            logger.info("Add run %s", run.name)
            with self._runs_lock:
                is_new = run.name not in self._runs
                self._runs[run.name] = run
                if is_new:
                    self._runs = OrderedDict(sorted(self._runs.items()))

                # Update is_active
                if not self._is_active:
                    self._is_active = True

    def _get_run_dirs(self):
        """Scan logdir, find PyTorch Profiler run directories.
        A directory is considered to be a run if it contains 1 or more *.pt.trace.json[.gz].
        E.g. there are 2 runs: run1, run2
            /run1
                /[worker1].pt.trace.json.gz
                /[worker2].pt.trace.json.gz
            /run2
                /[worker1].pt.trace.json
        """
        for root, _, files in os.walk(self.logdir):
            for file in files:
                if utils.is_chrome_trace_file(file):
                    run_dir = os.path.abspath(root)
                    if run_dir == self.logdir:
                        name = os.path.basename(run_dir)
                    else:
                        name = os.path.relpath(run_dir, self.logdir)
                    yield name, run_dir
                    break

    def get_run(self, name) -> Run:
        with self._runs_lock:
            return self._runs.get(name, None)

    @wrappers.Request.application
    def runs_route(self, request):
        with self._runs_lock:
            names = list(self._runs.keys())
        return self.respond_as_json(names)

    @wrappers.Request.application
    def views_route(self, request):
        name = request.args.get("run")
        run = self.get_run(name)
        views = sorted(run.views, key=lambda x: x.id)
        views_list = []
        for view in views:
            views_list.append(view.display_name)
        return self.respond_as_json(views_list)

    @wrappers.Request.application
    def workers_route(self, request):
        name = request.args.get("run")
        run = self.get_run(name)
        return self.respond_as_json(run.workers)

    @wrappers.Request.application
    def overview_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        run = self.get_run(name)
        profile = run.get_profile(worker)
        data = profile.overview
        data["environments"] = [{"title": "Number of Worker(s)", "value": str(len(run.workers))},
                                {"title": "Device Type", "value": "GPU" if profile.is_gpu_used else "CPU"}]
        if profile.is_gpu_used:
            data["environments"].append({"title": "Number of Device(s)", "value": "1"})
        return self.respond_as_json(data)

    @wrappers.Request.application
    def operation_pie_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        run = self.get_run(name)
        profile = run.get_profile(worker)
        if group_by == "OperationAndInputShape":
            return self.respond_as_json(profile.operation_pie_by_name_input)
        else:
            return self.respond_as_json(profile.operation_pie_by_name)

    @wrappers.Request.application
    def operation_table_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        run = self.get_run(name)
        profile = run.get_profile(worker)
        if group_by == "OperationAndInputShape":
            return self.respond_as_json(profile.operation_table_by_name_input)
        else:
            return self.respond_as_json(profile.operation_table_by_name)

    @wrappers.Request.application
    def kernel_pie_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        run = self.get_run(name)
        profile = run.get_profile(worker)
        return self.respond_as_json(profile.kernel_pie)

    @wrappers.Request.application
    def kernel_table_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        run = self.get_run(name)
        profile = run.get_profile(worker)
        if group_by == "Kernel":
            return self.respond_as_json(profile.kernel_table)
        else:
            return self.respond_as_json(profile.kernel_op_table)

    @wrappers.Request.application
    def trace_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")

        run = self.get_run(name)
        profile = run.get_profile(worker)
        fopen = open
        with fopen(profile.trace_file_path, 'rb') as f:
            raw_data = f.read()
        if profile.trace_file_path.endswith('.gz'):
            headers = []
            headers.append(('Content-Encoding', 'gzip'))
            return werkzeug.Response(raw_data, content_type="application/json", headers=headers)
        else:
            return werkzeug.Response(raw_data, content_type="application/json")

    @wrappers.Request.application
    def static_file_route(self, request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.css':
            mimetype = 'text/css'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'
        filepath = os.path.join(os.path.dirname(__file__), 'static', filename)
        try:
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError:
            return werkzeug.Response('404 Not Found', 'text/plain', code=404)
        return werkzeug.Response(
            contents, content_type=mimetype
        )

    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return werkzeug.Response(content, content_type="application/json")


def _load_run(queue, name, run_dir):
    import absl.logging
    absl.logging.use_absl_handler()

    try:
        logger.info("Load run %s", name)
        # Currently, assume run data is immutable, so just load once
        loader = RunLoader(name, run_dir)
        run = loader.load()
        logger.info("Run %s loaded", name)
        queue.put(run)
    except Exception as ex:
        logger.warning("Failed to load run %s. Exception=%s", ex, name, exc_info=True)
