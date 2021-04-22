import json
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

        host='localhost'
        port=6006

        link_prefix = 'http://{}:{}/data/plugin/pytorch_profiler/'.format(host, port)
        run_link = link_prefix + 'runs'
        expected_runs = b'["resnet50_num_workers_0", "resnet50_num_workers_4"]'

        expected_links_format=[
            link_prefix + 'overview?run={}&worker=worker0&view=Overview',
            link_prefix + 'operation?run={}&worker=worker0&view=Operator&group_by=Operation',
            link_prefix + 'operation/table?run={}&worker=worker0&view=Operator&group_by=Operation',
            link_prefix + 'kernel/table?run={}&worker=worker0&view=Kernel&group_by=Kernel',
            link_prefix + 'kernel?run={}&worker=worker0&view=Kernel&group_by=Kernel'
        ]

        tb = Popen(['tensorboard', '--logdir='+test_folder, '--port='+str(port)])

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
                if timeout % 10 == 0:
                    print("receive mismatched data, retrying", response.read())
                time.sleep(2)
                timeout -= 1
                if timeout<0:
                    tb.kill()
                    raise RuntimeError("Load run timeout")
            except Exception:
                continue

        links=[]
        for run in json.loads(expected_runs):
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

