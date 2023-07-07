import json
import os
import random
import shutil
import socket
import tempfile
import time
import unittest
import urllib
import urllib.request
from subprocess import Popen
from urllib.error import HTTPError


def get_samples_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '../samples')


class TestEnd2End(unittest.TestCase):

    # def test_tensorboard_gs(self):
    #    test_folder = 'gs://pe-tests-public/tb_samples/'
    #    expected_runs = b'["resnet50_profiler_api_num_workers_0", "resnet50_profiler_api_num_workers_4"]'
    #    self._test_tensorboard_with_arguments(test_folder, expected_runs, {'TORCH_PROFILER_START_METHOD':'spawn'})

    def test_tensorboard_end2end(self):
        test_folder = get_samples_dir()
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'

        print('starting spawn mode testing...')
        self._test_tensorboard_with_arguments(test_folder, expected_runs, {'TORCH_PROFILER_START_METHOD': 'spawn'})

    @unittest.skip('fork is not use anymore')
    def test_tensorboard_fork(self):
        test_folder = get_samples_dir()
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'

        print('starting fork mode testing')
        self._test_tensorboard_with_arguments(test_folder, expected_runs)

    def test_tensorboard_with_path_prefix(self):
        test_folder = get_samples_dir()
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'
        self._test_tensorboard_with_arguments(test_folder, expected_runs, path_prefix='/tensorboard/viewer/')

    def test_tensorboard_with_symlinks(self):
        logdir = tempfile.mkdtemp(prefix='tensorboard_logdir')

        samples_dir = get_samples_dir()

        # Create the following layout, with 1 symlink to a run dir, and 1 regular run dir:
        # logdir/
        #     run_concrete/
        #     run_symlink/ --> path/to/samples/resnet50_num_workers_4/
        shutil.copytree(os.path.join(samples_dir, 'resnet50_num_workers_0'), os.path.join(logdir, 'run_concrete'))
        os.symlink(os.path.join(samples_dir, 'resnet50_num_workers_4'), os.path.join(logdir, 'run_symlink'))

        expected_runs = b'["run_concrete", "run_symlink"]'
        self._test_tensorboard_with_arguments(logdir, expected_runs)

        shutil.rmtree(logdir)

    def _test_tensorboard_with_arguments(self, test_folder, expected_runs, env=None, path_prefix=None):
        host = 'localhost'
        port = random.randint(6008, 65535)

        try:
            if env:
                env_copy = os.environ.copy()
                env_copy.update(env)
                env = env_copy
            if not path_prefix:
                tb = Popen(['tensorboard', '--logdir='+test_folder, '--port='+str(port)], env=env)
            else:
                tb = Popen(['tensorboard', '--logdir='+test_folder, '--port='+str(port),
                           '--path_prefix='+path_prefix], env=env)
            self._test_tensorboard(host, port, expected_runs, path_prefix)
        finally:
            pid = tb.pid
            print('tensorboard process {} is terminating.'.format(pid))
            tb.terminate()

    def _test_tensorboard(self, host, port, expected_runs, path_prefix):
        if not path_prefix:
            link_prefix = 'http://{}:{}/data/plugin/pytorch_profiler/'.format(host, port)
        else:
            path_prefix = path_prefix.strip('/')
            link_prefix = 'http://{}:{}/{}/data/plugin/pytorch_profiler/'.format(host, port, path_prefix)
        run_link = link_prefix + 'runs'

        expected_links_format = [
            link_prefix + 'overview?run={}&worker=worker0&span=1&view=Overview',
            link_prefix + 'operation?run={}&worker=worker0&span=1&view=Operator&group_by=Operation',
            link_prefix + 'operation/table?run={}&worker=worker0&span=1&view=Operator&group_by=Operation',
            link_prefix + 'kernel/table?run={}&worker=worker0&span=1&view=Kernel&group_by=Kernel',
            link_prefix + 'kernel?run={}&worker=worker0&span=1&view=Kernel&group_by=Kernel'
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
                    self.fail('tensorboard start timeout')
                continue

        retry_times = 60

        while True:
            try:
                response = urllib.request.urlopen(run_link)
                data = response.read()
                runs = None
                if data:
                    data = json.loads(data)
                    runs = data.get('runs')
                    if runs:
                        runs = '[{}]'.format(', '.join(['"{}"'.format(i) for i in runs]))
                        runs = runs.encode('utf-8')
                if runs == expected_runs:
                    break
                if retry_times % 10 == 0:
                    print('receive mismatched data, retrying', data)
                time.sleep(2)
                retry_times -= 1
                if retry_times < 0:
                    self.fail('Load run timeout')
            except Exception:
                if retry_times > 0:
                    continue
                else:
                    raise

        links = []
        for run in json.loads(expected_runs):
            for expected_link in expected_links_format:
                links.append(expected_link.format(run))

        if os.environ.get('TORCH_PROFILER_REGEN_RESULT_CHECK') == '1':
            with open('result_check_file.txt', 'w', encoding='utf-8') as f:
                # NOTE: result_check_file.txt is manually generated and verified.
                # And then checked-in so that we can make sure that frontend
                # content change can be detected on code change.
                for link in links:
                    response = urllib.request.urlopen(link)
                    f.write(response.read().decode('utf-8'))
                    f.write('\n')
        else:
            with open('result_check_file.txt', 'r') as f:
                lines = f.readlines()
                i = 0
                print('starting testing...')
                for link in links:
                    try:
                        response = urllib.request.urlopen(link)
                        self.assertEqual(response.read(), lines[i].strip().encode(encoding='utf-8'))
                        i = i + 1
                    except HTTPError as e:
                        self.fail(e)
            self.assertEqual(i, 10)
            print('ending testing...')


if __name__ == '__main__':
    unittest.main()
