# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import atexit
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import gzip
from collections import OrderedDict

import werkzeug
from tensorboard.plugins import base_plugin
from werkzeug import wrappers

from . import consts, io, utils
from .profiler import RunLoader
from .run import Run

logger = utils.get_logger()


class TorchProfilerPlugin(base_plugin.TBPlugin):
    """TensorBoard plugin for Torch Profiler."""

    plugin_name = consts.PLUGIN_NAME
    headers = [('X-Content-Type-Options', 'nosniff')]

    def __init__(self, context):
        """Instantiates TorchProfilerPlugin.
        Args:
          context: A base_plugin.TBContext instance.
        """
        super(TorchProfilerPlugin, self).__init__(context)
        start_method = os.getenv('TORCH_PROFILER_START_METHOD')
        if start_method:
            mp.set_start_method(start_method, force=True)
        self.logdir = io.abspath(context.logdir.rstrip('/'))

        self._is_active = None
        self._is_active_initialized_event = threading.Event()

        self._runs = OrderedDict()
        self._runs_lock = threading.Lock()

        self._cache = io.Cache()
        self._queue = mp.Queue()
        monitor_runs = threading.Thread(target=self._monitor_runs, name="monitor_runs", daemon=True)
        monitor_runs.start()

        receive_runs = threading.Thread(target=self._receive_runs, name="receive_runs", daemon=True)
        receive_runs.start()

        def clean():
            logger.debug("starting cleanup...")
            self._cache.__exit__(*sys.exc_info())
        atexit.register(clean)

    def __getstate__(self):
        '''The multiprocessing module can start one of three ways: spawn, fork, or forkserver. 
        The default mode is fork in Unix and spawn on Windows and macOS.
        Therefore, the __getstate__ and __setstate__ are used to pickle/unpickle the state in spawn mode.
        '''
        data = self.__dict__.copy()
        # remove self._runs_lock and self._is_active_initialized_event since they are threading stuff to
        # make sure the plugin instance could be pickled to the data parsing process
        # otherwise, 'TypeError: cannot pickle '_thread.lock' object' will be raised.
        del data['_runs_lock']
        del data['_is_active_initialized_event']
        logger.debug("TorchProfilerPlugin.__getstate__: %s " % data)
        return data

    def __setstate__(self, d):
        '''The default logging level in new process is warning. 
        As the result, the logger.info will be ignored. We have to leverage the multiprocessing.get_logger() which will be used by the 
        python multiprocessing.
        '''
        with utils.mp_logging() as logger:
            logger.debug("TorchProfilerPlugin.__setstate__ with %s " % d)
        self.__dict__.update(d)

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.
        """
        self._is_active_initialized_event.wait()
        return self._is_active

    def get_plugin_apps(self):
        return {
            "/index.js": self.static_file_route,
            "/index.html": self.static_file_route,
            "/trace_viewer_full.html": self.static_file_route,
            "/trace_embedding.html": self.static_file_route,
            "/runs": self.runs_route,
            "/views": self.views_route,
            "/workers": self.workers_route,
            "/overview": self.overview_route,
            "/operation": self.operation_pie_route,
            "/operation/table": self.operation_table_route,
            "/operation/stack": self.operation_stack_route,
            "/kernel": self.kernel_pie_route,
            "/kernel/table": self.kernel_table_route,
            "/trace": self.trace_route,
            "/distributed/overlap": self.comm_overlap_route,
            "/distributed/waittime": self.comm_wait_route,
            "/distributed/commops": self.comm_ops_route
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js")

    @wrappers.Request.application
    def runs_route(self, request):
        with self._runs_lock:
            names = list(self._runs.keys())
        return self.respond_as_json(names)

    @wrappers.Request.application
    def views_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        run = self._get_run(name)
        profile = run.get_profile(worker)
        views = sorted(profile.views, key=lambda x: x.id)
        views_list = []
        for view in views:
            views_list.append(view.display_name)
        return self.respond_as_json(views_list)

    @wrappers.Request.application
    def workers_route(self, request):
        name = request.args.get("run")
        run = self._get_run(name)
        return self.respond_as_json(run.workers)

    @wrappers.Request.application
    def overview_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        run = self._get_run(name)
        profile = run.get_profile(worker)
        data = profile.overview
        is_gpu_used = profile.has_runtime or profile.has_kernel or profile.has_memcpy_or_memset
        data["environments"] = [{"title": "Number of Worker(s)", "value": str(len(run.workers))},
                                {"title": "Device Type", "value": "GPU" if is_gpu_used else "CPU"}]
        for gpu_id in profile.gpu_ids:
            data["environments"].append({"title": "GPU Utilization of GPU{}".format(gpu_id),
                                         "value": "{} %".format(round(profile.gpu_utilization[gpu_id] * 100, 2))})
            if profile.sm_efficency[gpu_id] > 0.0:
                data["environments"].append({"title": "Est. SM Efficiency of GPU{}".format(gpu_id),
                                             "value": "{} %".format(round(profile.sm_efficency[gpu_id] * 100, 2))})
            if profile.occupancy[gpu_id] > 0.0:
                data["environments"].append({"title": "Est. Achieved Occupancy of GPU{}".format(gpu_id),
                                             "value": "{} %".format(round(profile.occupancy[gpu_id], 2))})
        return self.respond_as_json(data)

    @wrappers.Request.application
    def operation_pie_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        run = self._get_run(name)
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
        run = self._get_run(name)
        profile = run.get_profile(worker)
        if group_by == "OperationAndInputShape":
            return self.respond_as_json(profile.operation_table_by_name_input)
        else:
            return self.respond_as_json(profile.operation_table_by_name)

    @wrappers.Request.application
    def operation_stack_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        op_name = request.args.get("op_name")
        input_shape = request.args.get("input_shape")
        run = self._get_run(name)
        profile = run.get_profile(worker)
        if group_by == "OperationAndInputShape":
            return self.respond_as_json(profile.operation_stack_by_name_input[str(op_name)+"###"+str(input_shape)])
        else:
            return self.respond_as_json(profile.operation_stack_by_name[str(op_name)])

    @wrappers.Request.application
    def kernel_pie_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        run = self._get_run(name)
        profile = run.get_profile(worker)
        return self.respond_as_json(profile.kernel_pie)

    @wrappers.Request.application
    def kernel_table_route(self, request):
        name = request.args.get("run")
        worker = request.args.get("worker")
        group_by = request.args.get("group_by")
        run = self._get_run(name)
        profile = run.get_profile(worker)
        if group_by == "Kernel":
            return self.respond_as_json(profile.kernel_table)
        else:
            return self.respond_as_json(profile.kernel_op_table)

    @wrappers.Request.application
    def trace_route(self, request):
        def build_trace_counter_gpu_util(gpu_id, start_time, counter_value):
            util_json = ", {{\"ph\":\"C\", \"name\":\"GPU {} Utilization\", " \
                        "\"pid\":{}, \"ts\":{}, " \
                        "\"args\":{{\"GPU Utilization\":{}}}}}".format(
                gpu_id, gpu_id, start_time, counter_value
            )
            return util_json

        def build_trace_counter_sm_efficiency(gpu_id, start_time, counter_value):
            util_json = ", {{\"ph\":\"C\", \"name\":\"GPU {} Est. SM Efficiency\", " \
                        "\"pid\":{}, \"ts\":{}, " \
                        "\"args\":{{\"Est. SM Efficiency\":{}}}}}".format(
                gpu_id, gpu_id, start_time, counter_value
            )
            return util_json

        name = request.args.get("run")
        worker = request.args.get("worker")

        run = self._get_run(name)
        profile = run.get_profile(worker)
        raw_data, includes_gpu_metrics = self._cache.read(profile.trace_file_path)

        if not profile.has_kernel:  # Pure CPU.
            if not profile.trace_file_path.endswith('.gz'):
                raw_data = gzip.compress(raw_data, 1)
        elif not includes_gpu_metrics:
            counter_json_str = ""

            for gpu_id in range(len(profile.gpu_util_buckets)):
                buckets = profile.gpu_util_buckets[gpu_id]
                for b in buckets:
                    json_str = build_trace_counter_gpu_util(gpu_id, b[0], b[1])
                    counter_json_str += json_str

            for gpu_id in range(len(profile.approximated_sm_efficency_ranges)):
                ranges = profile.approximated_sm_efficency_ranges[gpu_id]
                for r in ranges:
                    efficiency_json_start = build_trace_counter_sm_efficiency(gpu_id, r[0][0], r[1])
                    efficiency_json_finish = build_trace_counter_sm_efficiency(gpu_id, r[0][1], 0)
                    counter_json_str += (efficiency_json_start + efficiency_json_finish)

            counter_json_bytes = bytes(counter_json_str, 'utf-8')
            if profile.trace_file_path.endswith('.gz'):
                raw_data = gzip.decompress(raw_data)
            raw_data = b''.join([raw_data[:-2], counter_json_bytes, b']}'])

            raw_data = gzip.compress(raw_data, 1)
            self._cache.write_gpu_metrics(raw_data, profile.trace_file_path)
        headers = []
        headers.append(('Content-Encoding', 'gzip'))
        headers.extend(TorchProfilerPlugin.headers)
        return werkzeug.Response(raw_data, content_type="application/json", headers=headers)

    @wrappers.Request.application
    def comm_overlap_route(self, request):
        name = request.args.get("run")
        run = self._get_run(name)
        profile = run.get_profile("All")
        return self.respond_as_json(profile.steps_to_overlap)

    @wrappers.Request.application
    def comm_wait_route(self, request):
        name = request.args.get("run")
        run = self._get_run(name)
        profile = run.get_profile("All")
        return self.respond_as_json(profile.steps_to_wait)

    @wrappers.Request.application
    def comm_ops_route(self, request):
        name = request.args.get("run")
        run = self._get_run(name)
        profile = run.get_profile("All")
        return self.respond_as_json(profile.comm_ops)

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
            return werkzeug.Response('404 Not Found', 'text/plain', code=404, headers=TorchProfilerPlugin.headers)
        return werkzeug.Response(
            contents, content_type=mimetype, headers=TorchProfilerPlugin.headers
        )

    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return werkzeug.Response(content, content_type="application/json", headers=TorchProfilerPlugin.headers)

    def _monitor_runs(self):
        logger.info("Monitor runs begin")

        try:
            touched = set()
            while True:
                try:
                    logger.debug("Scan run dir")
                    run_dirs = self._get_run_dirs()

                    # Assume no deletion on run directories, trigger async load if find a new run
                    for run_dir in run_dirs:
                        # Set _is_active quickly based on file pattern match, don't wait for data loading
                        if not self._is_active:
                            self._is_active = True
                            self._is_active_initialized_event.set()

                        if run_dir not in touched:
                            touched.add(run_dir)
                            logger.info("Find run directory %s", run_dir)
                            # Use multiprocessing to avoid UI stall and reduce data parsing time
                            process = mp.Process(target=self._load_run, args=(run_dir,))
                            process.daemon = True
                            process.start()
                except Exception as ex:
                    logger.warning("Failed to scan runs. Exception=%s", ex, exc_info=True)

                time.sleep(consts.MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS)
        except:
            logger.exception("Failed to start monitor_runs")

    def _receive_runs(self):
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
                    self._is_active_initialized_event.set()

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
        for root, _, files in io.walk(self.logdir):
            for file in files:
                if utils.is_chrome_trace_file(file):
                    yield root
                    break

    def _load_run(self, run_dir):
        import absl.logging
        absl.logging.use_absl_handler()

        try:
            name = self._get_run_name(run_dir)
            logger.info("Load run %s", name)
            # Currently, assume run data is immutable, so just load once
            loader = RunLoader(name, run_dir, self._cache)
            run = loader.load()
            logger.info("Run %s loaded", name)
            self._queue.put(run)
        except Exception as ex:
            logger.warning("Failed to load run %s. Exception=%s", ex, name, exc_info=True)

    def _get_run(self, name) -> Run:
        with self._runs_lock:
            return self._runs.get(name, None)

    def _get_run_name(self, run_dir):
        logdir = io.abspath(self.logdir)
        if run_dir == logdir:
            name = io.basename(run_dir)
        else:
            name = io.relpath(run_dir, logdir)
        return name
