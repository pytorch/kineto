import os
import socket
import time
import urllib
import urllib.request
import unittest
from subprocess import Popen


class TestEnd2End(unittest.TestCase):

    def test_tensorboard_end2end(self):
        test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../samples')
        tb = Popen(['tensorboard', '--logdir='+test_folder])

        run_link = "http://localhost:6006/data/plugin/pytorch_profiler/runs"
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'
        host='localhost'
        port=6006

        timeout = 60
        while True:
            try:
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                print('tensorboard start successfully')
                break
            except socket.error:
                time.sleep(2)
                timeout -= 1
                if timeout < 0:
                    tb.kill()
                    raise RuntimeError("tensorboard start timeout")
                continue

        timeout = 60
        while True:
            try:
                response = urllib.request.urlopen(run_link)
                if response.read()==expected_runs:
                    break
                time.sleep(2)
                timeout -= 1
                if timeout<0:
                    tb.kill()
                    raise RuntimeError("Load run timeout")
            except Exception:
                continue

        link_prefix = 'http://localhost:6006/data/plugin/pytorch_profiler/'
        expected_links_format=[]
        expected_links_format.append(link_prefix + 'overview?run={}&worker=worker0&view=Overview')
        expected_links_format.append(link_prefix + 'operation?run={}&worker=worker0&view=Operator&group_by=Operation')
        expected_links_format.append(link_prefix + 'operation/table?run={}&worker=worker0&view=Operator&group_by=Operation')
        expected_links_format.append(link_prefix + 'kernel/table?run={}&worker=worker0&view=Kernel&group_by=Kernel')
        expected_links_format.append(link_prefix + 'kernel?run={}&worker=worker0&view=Kernel&group_by=Kernel')
        links=[]
        for run in ["resnet50_num_workers_0",
                    "resnet50_num_workers_4"]:
            for expected_link in expected_links_format:
                links.append(expected_link.format(run))

        try:
            with open('result_check_file.txt', 'r') as f:
                lines=f.readlines()
                i = 0
                for link in links:
                    response = urllib.request.urlopen(link)
                    self.assertEqual(response.read(), lines[i].strip().encode(encoding="utf-8"))
                    i = i + 1
            self.assertEqual(i, 10)
        finally:
            tb.kill()

