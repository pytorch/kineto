# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import atexit
import gzip
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from queue import Queue

import werkzeug
from tensorboard.plugins import base_plugin
from werkzeug import exceptions, wrappers

from . import consts, io, utils
from .profiler import RunLoader
from .run import DistributedRunProfile, Run, RunProfile

logger = utils.get_logger()


def decorate_headers(func):
    def wrapper(*args, **kwargs):
        headers = func(*args, **kwargs)
        headers.extend(TorchProfilerPlugin.headers)
        return headers
    return wrapper


exceptions.HTTPException.get_headers = decorate_headers(exceptions.HTTPException.get_headers)


class TorchProfilerPlugin(base_plugin.TBPlugin):
    """TensorBoard plugin for Torch Profiler."""

    plugin_name = consts.PLUGIN_NAME
    headers = [('X-Content-Type-Options', 'nosniff')]
    CONTENT_TYPE = 'application/json'

    def __init__(self, context: base_plugin.TBContext):
        """Instantiates TorchProfilerPlugin.
        Args:
          context: A base_plugin.TBContext instance.
        """
        super(TorchProfilerPlugin, self).__init__(context)
        if not context.logdir and context.flags.logdir_spec:
            dirs = context.flags.logdir_spec.split(',')
            if len(dirs) > 1:
                logger.warning(f"Multiple directories are specified by --logdir_spec flag. TorchProfilerPlugin will load the first one: \n {dirs[0]}")
            self.logdir = io.abspath(dirs[0].rstrip('/'))
        else:
            self.logdir = io.abspath(context.logdir.rstrip('/'))

        self._load_lock = threading.Lock()
        self._load_threads = []

        self._runs = OrderedDict()
        self._runs_lock = threading.Lock()

        self._temp_dir = tempfile.mkdtemp()
        self._cache = io.Cache(self._temp_dir)
        self._queue = Queue()
        self._gpu_metrics_file_dict = {}
        monitor_runs = threading.Thread(target=self._monitor_runs, name='monitor_runs', daemon=True)
        monitor_runs.start()

        receive_runs = threading.Thread(target=self._receive_runs, name='receive_runs', daemon=True)
        receive_runs.start()

        self.diff_run_cache = {}
        self.diff_run_flatten_cache = {}

        def clean():
            logger.debug('starting cleanup...')
            self._cache.__exit__(*sys.exc_info())
            logger.debug('remove temporary cache directory %s' % self._temp_dir)
            shutil.rmtree(self._temp_dir)

        atexit.register(clean)

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.
        If there is no any pending run, hide the plugin
        """
        if self.is_loading:
            return True
        else:
            with self._runs_lock:
                return bool(self._runs)

    def get_plugin_apps(self):
        return {
            '/index.js': self.static_file_route,
            '/index.html': self.static_file_route,
            '/trace_viewer_full.html': self.static_file_route,
            '/trace_embedding.html': self.static_file_route,
            '/runs': self.runs_route,
            '/views': self.views_route,
            '/workers': self.workers_route,
            '/spans': self.spans_route,
            '/overview': self.overview_route,
            '/operation': self.operation_pie_route,
            '/operation/table': self.operation_table_route,
            '/operation/stack': self.operation_stack_route,
            '/kernel': self.kernel_pie_route,
            '/kernel/table': self.kernel_table_route,
            '/kernel/tc_pie': self.kernel_tc_route,
            '/trace': self.trace_route,
            '/distributed/gpuinfo': self.dist_gpu_info_route,
            '/distributed/overlap': self.comm_overlap_route,
            '/distributed/waittime': self.comm_wait_route,
            '/distributed/commops': self.comm_ops_route,
            '/memory': self.memory_route,
            '/memory_curve': self.memory_curve_route,
            '/memory_events': self.memory_events_route,
            '/module': self.module_route,
            '/tree': self.op_tree_route,
            '/diff': self.diff_run_route,
            '/diffnode': self.diff_run_node_route,
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path='/index.js', disable_reload=True)

    @wrappers.Request.application
    def runs_route(self, request: werkzeug.Request):
        with self._runs_lock:
            names = list(self._runs.keys())

        data = {
            'runs': names,
            'loading': self.is_loading
        }
        return self.respond_as_json(data)

    @wrappers.Request.application
    def views_route(self, request: werkzeug.Request):
        name = request.args.get('run')
        self._validate(run=name)
        run = self._get_run(name)
        views_list = [view.display_name for view in run.views]
        return self.respond_as_json(views_list)

    @wrappers.Request.application
    def workers_route(self, request: werkzeug.Request):
        name = request.args.get('run')
        view = request.args.get('view')
        self._validate(run=name, view=view)
        run = self._get_run(name)
        return self.respond_as_json(run.get_workers(view))

    @wrappers.Request.application
    def spans_route(self, request: werkzeug.Request):
        name = request.args.get('run')
        worker = request.args.get('worker')
        self._validate(run=name, worker=worker)
        run = self._get_run(name)
        return self.respond_as_json(run.get_spans(worker))

    @wrappers.Request.application
    def overview_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        name = request.args.get('run')
        run = self._get_run(name)
        data = profile.overview
        is_gpu_used = profile.has_runtime or profile.has_kernel or profile.has_memcpy_or_memset
        normal_workers = [worker for worker in run.workers if worker != 'All']
        data['environments'] = [{'title': 'Number of Worker(s)', 'value': str(len(normal_workers))},
                                {'title': 'Device Type', 'value': 'GPU' if is_gpu_used else 'CPU'}]
        if profile.gpu_summary and profile.gpu_tooltip:
            data['gpu_metrics'] = {'title': 'GPU Summary',
                                   'data': profile.gpu_summary,
                                   'tooltip': profile.gpu_tooltip}

        return self.respond_as_json(data)

    @wrappers.Request.application
    def operation_pie_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        group_by = request.args.get('group_by')
        if group_by == 'OperationAndInputShape':
            return self.respond_as_json(profile.operation_pie_by_name_input)
        else:
            return self.respond_as_json(profile.operation_pie_by_name)

    @wrappers.Request.application
    def operation_table_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        group_by = request.args.get('group_by')
        if group_by == 'OperationAndInputShape':
            return self.respond_as_json(profile.operation_table_by_name_input)
        else:
            return self.respond_as_json(profile.operation_table_by_name)

    @wrappers.Request.application
    def operation_stack_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        op_name = request.args.get('op_name')
        self._validate(op_name=op_name)
        group_by = request.args.get('group_by')
        input_shape = request.args.get('input_shape')
        if group_by == 'OperationAndInputShape':
            return self.respond_as_json(profile.operation_stack_by_name_input[str(op_name)+'###'+str(input_shape)])
        else:
            return self.respond_as_json(profile.operation_stack_by_name[str(op_name)])

    @wrappers.Request.application
    def kernel_pie_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        return self.respond_as_json(profile.kernel_pie)

    @wrappers.Request.application
    def kernel_table_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        group_by = request.args.get('group_by')
        if group_by == 'Kernel':
            return self.respond_as_json(profile.kernel_table)
        else:
            return self.respond_as_json(profile.kernel_op_table)

    @wrappers.Request.application
    def kernel_tc_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        return self.respond_as_json(profile.tc_pie)

    @wrappers.Request.application
    def trace_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)

        if not profile.has_kernel:  # Pure CPU.
            raw_data = self._cache.read(profile.trace_file_path)
            if not profile.trace_file_path.endswith('.gz'):
                raw_data = gzip.compress(raw_data, 1)
        else:
            file_with_gpu_metrics = self._gpu_metrics_file_dict.get(profile.trace_file_path)
            if file_with_gpu_metrics:
                raw_data = io.read(file_with_gpu_metrics)
            else:
                raw_data = self._cache.read(profile.trace_file_path)
                if profile.trace_file_path.endswith('.gz'):
                    raw_data = gzip.decompress(raw_data)
                raw_data = profile.append_gpu_metrics(raw_data)

                # write the data to temp file
                fp = tempfile.NamedTemporaryFile('w+b', suffix='.json.gz', dir=self._temp_dir, delete=False)
                fp.close()
                # Already compressed, no need to gzip.open
                with open(fp.name, mode='wb') as file:
                    file.write(raw_data)
                self._gpu_metrics_file_dict[profile.trace_file_path] = fp.name

        headers = [('Content-Encoding', 'gzip')]
        headers.extend(TorchProfilerPlugin.headers)
        return werkzeug.Response(raw_data, content_type=TorchProfilerPlugin.CONTENT_TYPE, headers=headers)

    @wrappers.Request.application
    def dist_gpu_info_route(self, request: werkzeug.Request):
        profile = self._get_distributed_profile_for_request(request)
        return self.respond_as_json(profile.gpu_info)

    @wrappers.Request.application
    def comm_overlap_route(self, request: werkzeug.Request):
        profile = self._get_distributed_profile_for_request(request)
        return self.respond_as_json(profile.steps_to_overlap)

    @wrappers.Request.application
    def comm_wait_route(self, request: werkzeug.Request):
        profile = self._get_distributed_profile_for_request(request)
        return self.respond_as_json(profile.steps_to_wait)

    @wrappers.Request.application
    def comm_ops_route(self, request: werkzeug.Request):
        profile = self._get_distributed_profile_for_request(request)
        return self.respond_as_json(profile.comm_ops)

    @wrappers.Request.application
    def memory_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        start_ts = request.args.get('start_ts', None)
        end_ts = request.args.get('end_ts', None)
        memory_metric = request.args.get('memory_metric', 'KB')
        if start_ts is not None:
            start_ts = int(start_ts)
        if end_ts is not None:
            end_ts = int(end_ts)

        return self.respond_as_json(
            profile.get_memory_stats(start_ts=start_ts, end_ts=end_ts, memory_metric=memory_metric), True)

    @wrappers.Request.application
    def memory_curve_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        time_metric = request.args.get('time_metric', 'ms')
        memory_metric = request.args.get('memory_metric', 'MB')
        return self.respond_as_json(
            profile.get_memory_curve(time_metric=time_metric, memory_metric=memory_metric), True)

    @wrappers.Request.application
    def memory_events_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        start_ts = request.args.get('start_ts', None)
        end_ts = request.args.get('end_ts', None)
        time_metric = request.args.get('time_metric', 'ms')
        memory_metric = request.args.get('memory_metric', 'KB')
        if start_ts is not None:
            start_ts = int(start_ts)
        if end_ts is not None:
            end_ts = int(end_ts)

        return self.respond_as_json(
            profile.get_memory_events(start_ts, end_ts, time_metric=time_metric,
                                      memory_metric=memory_metric), True)

    @wrappers.Request.application
    def module_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        content = profile.get_module_view()
        if content:
            return self.respond_as_json(content, True)
        else:
            name = request.args.get('run')
            worker = request.args.get('worker')
            span = request.args.get('span')
            raise exceptions.NotFound('could not find the run for %s/%s/%s' % (name, worker, span))

    @wrappers.Request.application
    def op_tree_route(self, request: werkzeug.Request):
        profile = self._get_profile_for_request(request)
        content = profile.get_operator_tree()
        return self.respond_as_json(content, True)

    @wrappers.Request.application
    def diff_run_route(self, request: werkzeug.Request):
        base, exp = self.get_diff_runs(request)
        diff_stats = self.get_diff_status(base, exp)
        content = diff_stats.get_diff_tree_summary()
        return self.respond_as_json(content, True)

    @wrappers.Request.application
    def diff_run_node_route(self, request: werkzeug.Request):
        base, exp = self.get_diff_runs(request)
        path = request.args.get('path', '0')
        stats_dict = self.get_diff_stats_dict(base, exp)
        diff_stat = stats_dict.get(path)
        if diff_stat is None:
            raise exceptions.NotFound('could not find diff run for %s' % (path))
        content = diff_stat.get_diff_node_summary(path)
        return self.respond_as_json(content, True)

    @wrappers.Request.application
    def static_file_route(self, request: werkzeug.Request):
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
            raise exceptions.NotFound('404 Not Found')
        return werkzeug.Response(
            contents, content_type=mimetype, headers=TorchProfilerPlugin.headers
        )

    @staticmethod
    def respond_as_json(obj, compress: bool = False):
        content = json.dumps(obj)
        headers = []
        headers.extend(TorchProfilerPlugin.headers)
        if compress:
            content_bytes = content.encode('utf-8')
            raw_data = gzip.compress(content_bytes, 1)
            headers.append(('Content-Encoding', 'gzip'))
            return werkzeug.Response(raw_data, content_type=TorchProfilerPlugin.CONTENT_TYPE, headers=headers)
        else:
            return werkzeug.Response(content, content_type=TorchProfilerPlugin.CONTENT_TYPE, headers=headers)

    @property
    def is_loading(self):
        with self._load_lock:
            return bool(self._load_threads)

    def get_diff_runs(self, request: werkzeug.Request):
        name = request.args.get('run')
        span = request.args.get('span')
        worker = request.args.get('worker')
        self._validate(run=name, worker=worker, span=span)
        base = self._get_profile(name, worker, span)

        exp_name = request.args.get('exp_run')
        exp_span = request.args.get('exp_span')
        exp_worker = request.args.get('exp_worker')
        self._validate(exp_run=exp_name, exp_worker=exp_worker, exp_span=exp_span)
        exp = self._get_profile(exp_name, exp_worker, exp_span)

        return base, exp

    def get_diff_status(self, base: RunProfile, exp: RunProfile):
        key = (base, exp)
        diff_stats = self.diff_run_cache.get(key)
        if diff_stats is None:
            diff_stats = base.compare_run(exp)
            self.diff_run_cache[key] = diff_stats

        return diff_stats

    def get_diff_stats_dict(self, base: RunProfile, exp: RunProfile):
        key = (base, exp)
        stats_dict = self.diff_run_flatten_cache.get(key)
        if stats_dict is None:
            diff_stats = self.get_diff_status(base, exp)
            stats_dict = diff_stats.flatten_diff_tree()
            self.diff_run_flatten_cache[key] = stats_dict
        return stats_dict

    def _monitor_runs(self):
        logger.info('Monitor runs begin')

        try:
            touched = set()
            while True:
                try:
                    logger.debug('Scan run dir')
                    run_dirs = self._get_run_dirs()

                    has_dir = False
                    # Assume no deletion on run directories, trigger async load if find a new run
                    for run_dir in run_dirs:
                        has_dir = True
                        if run_dir not in touched:
                            touched.add(run_dir)
                            logger.info('Find run directory %s', run_dir)
                            # Use threading to avoid UI stall and reduce data parsing time
                            t = threading.Thread(target=self._load_run, args=(run_dir,))
                            t.start()
                            with self._load_lock:
                                self._load_threads.append(t)

                    if not has_dir:
                        # handle directory removed case.
                        self._runs.clear()
                except Exception as ex:
                    logger.warning('Failed to scan runs. Exception=%s', ex, exc_info=True)

                time.sleep(consts.MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS)
        except Exception:
            logger.exception('Failed to start monitor_runs')

    def _receive_runs(self):
        while True:
            run: Run = self._queue.get()
            if run is None:
                continue

            logger.info('Add run %s', run.name)
            with self._runs_lock:
                is_new = run.name not in self._runs
                self._runs[run.name] = run
                if is_new:
                    self._runs = OrderedDict(sorted(self._runs.items()))

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
        if not io.isdir(self.logdir):
            return
        for root, _, files in io.walk(self.logdir):
            for file in files:
                if utils.is_chrome_trace_file(file):
                    yield root
                    break

    def _load_run(self, run_dir):
        try:
            name = self._get_run_name(run_dir)
            logger.info('Load run %s', name)
            # Currently, assume run data is immutable, so just load once
            loader = RunLoader(name, run_dir, self._cache)
            run = loader.load()
            logger.info('Run %s loaded', name)
            self._queue.put(run)
        except Exception as ex:
            logger.warning('Failed to load run %s. Exception=%s', ex, name, exc_info=True)

        t = threading.current_thread()
        with self._load_lock:
            try:
                self._load_threads.remove(t)
            except ValueError:
                logger.warning('could not find the thread {}'.format(run_dir))

    def _get_run(self, name) -> Run:
        with self._runs_lock:
            run = self._runs.get(name, None)

        if run is None:
            raise exceptions.NotFound('could not find the run for %s' % (name))

        return run

    def _get_run_name(self, run_dir):
        logdir = io.abspath(self.logdir)
        if run_dir == logdir:
            name = io.basename(run_dir)
        else:
            name = io.relpath(run_dir, logdir)
        return name

    def _get_profile_for_request(self, request: werkzeug.Request) -> RunProfile:
        name = request.args.get('run')
        span = request.args.get('span')
        worker = request.args.get('worker')
        self._validate(run=name, worker=worker)
        profile = self._get_profile(name, worker, span)
        if not isinstance(profile, RunProfile):
            raise exceptions.BadRequest('Get an unexpected profile type %s for %s/%s' % (type(profile), name, worker))

        return profile

    def _get_distributed_profile_for_request(self, request: werkzeug.Request) -> DistributedRunProfile:
        name = request.args.get('run')
        span = request.args.get('span')
        self._validate(run=name)
        profile = self._get_profile(name, 'All', span)
        if not isinstance(profile, DistributedRunProfile):
            raise exceptions.BadRequest('Get an unexpected distributed profile type %s for %s' % (type(profile), name))

        return profile

    def _get_profile(self, name, worker, span):
        run = self._get_run(name)
        profile = run.get_profile(worker, span)
        if profile is None:
            raise exceptions.NotFound('could not find the profile for %s/%s/%s ' % (name, worker, span))
        return profile

    def _validate(self, **kwargs):
        for name, v in kwargs.items():
            if v is None:
                raise exceptions.BadRequest('Must specify %s in request url' % (name))
