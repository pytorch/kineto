"""SimpleHTTPServer with IPv6 suport to run on TAP which will be IPV6 only.

Also provides /healthz support so that test can wait for the server to start.
"""

from __future__ import print_function

import http.server
import os
import socketserver

PORT = 4200

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
  os.chdir(os.getcwd() + "/frontend")
  print("Listening on port %s" % PORT)
  httpd.serve_forever()
