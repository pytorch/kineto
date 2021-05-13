import json
import os
import socket
import time
import urllib
import urllib.request
import unittest
from subprocess import Popen


class TestEnd2End(unittest.TestCase):

    def test_tensorboard_gs(self):
        test_folder = 'gs://pe-tests-public/tb_samples/'
        expected_runs = b'["resnet50_profiler_api_num_workers_0", "resnet50_profiler_api_num_workers_4"]'
        self._test_tensorboard_with_arguments(test_folder, expected_runs, {'TORCH_PROFILER_START_METHOD':'spawn'})

    def test_tensorboard_end2end(self):
        test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../samples')
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'

        print("starting fork mode testing")
        self._test_tensorboard_with_arguments(test_folder, expected_runs)
        print("starting spawn mode testing...")
        self._test_tensorboard_with_arguments(test_folder, expected_runs, {'TORCH_PROFILER_START_METHOD':'spawn'})

    def test_tensorboard_with_path_prefix(self):
        test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../samples')
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'
        self._test_tensorboard_with_arguments(test_folder, expected_runs, path_prefix='/tensorboard/viewer/')

    def _test_tensorboard_with_arguments(self, test_folder, expected_runs, env=None, path_prefix=None):
        host='localhost'
        port=6006

        try:
            if env:
                env_copy = os.environ.copy()
                env_copy.update(env)
                env = env_copy
            if not path_prefix:
                tb = Popen(['tensorboard', '--logdir='+test_folder, '--port='+str(port)], env=env)
            else:
                tb = Popen(['tensorboard', '--logdir='+test_folder, '--port='+str(port), '--path_prefix='+path_prefix], env=env)
            self._test_tensorboard(host, port, expected_runs, path_prefix)
        finally:
            pid = tb.pid
            tb.terminate()
            print("tensorboard process {} is terminated.".format(pid))

    def _test_tensorboard(self, host, port, expected_runs, path_prefix):
        if not path_prefix:
            link_prefix = 'http://{}:{}/data/plugin/pytorch_profiler/'.format(host, port)
        else:
            path_prefix = path_prefix.strip('/')
            link_prefix = 'http://{}:{}/{}/data/plugin/pytorch_profiler/'.format(host, port, path_prefix)
        run_link = link_prefix + 'runs'

        expected_links_format=[
            link_prefix + 'overview?run={}&worker=worker0&view=Overview',
            link_prefix + 'operation?run={}&worker=worker0&view=Operator&group_by=Operation',
            link_prefix + 'operation/table?run={}&worker=worker0&view=Operator&group_by=Operation',
            link_prefix + 'kernel/table?run={}&worker=worker0&view=Kernel&group_by=Kernel',
            link_prefix + 'kernel?run={}&worker=worker0&view=Kernel&group_by=Kernel'
        ]

        retry_times = 60
        while True:
            try:
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                print('tensorboard start successfully')
                break
            except socket.error:
                time.sleep(2)
                retry_times -= 1
                if retry_times < 0:
                    self.fail("tensorboard start timeout")
                continue

        retry_times = 60

        while True:
            try:
                response = urllib.request.urlopen(run_link)
                data = response.read()
                if data == expected_runs:
                    break
                if retry_times % 10 == 0:
                    print("receive mismatched data, retrying", data)
                time.sleep(2)
                retry_times -= 1
                if retry_times<0:
                    self.fail("Load run timeout")
            except Exception:
                if retry_times > 0:
                    continue
                else:
                    raise

        links=[]
        for run in json.loads(expected_runs):
            for expected_link in expected_links_format:
                links.append(expected_link.format(run))

        with open('result_check_file.txt', 'r') as f:
            lines=f.readlines()
            i = 0
            for link in links:
                response = urllib.request.urlopen(link)
                self.assertEqual(response.read(), lines[i].strip().encode(encoding="utf-8"))
                i = i + 1
        self.assertEqual(i, 10)
