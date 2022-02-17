#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2019-2021 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.# -*- coding: utf-8 -*-




import Pyro4
#import os
import socket
import inaFaceAnalyzer.commandline_utils as ifacu

#from argparse import Namespace
#import argparse.Namespace

description = '''Worker in charge of analyzing documents and receive orders
from a central job server. Workers need to have access to input files (url or path)
and must be allowed to write results in a centralized storage directory
To be used jointly with ina_face_analyzer_distributed_server.py .
The server must be run before runing the worker.
'''

hserver_uri = '''
The URI displayed by the server to be copy/pasted after runing program
ina_face_analyzer_distributed_server.py  .
Sample URI: PYRO:obj_25e3ee9d312848fcaf0784c2b80933c4@blahtop:44175
'''


if __name__ == '__main__':
    parser = ifacu.new_parser(description)

    parser.add_argument(dest='server_uri', help=hserver_uri)

    args = parser.parse_args()

    hostname = socket.gethostname()

    jobserver = Pyro4.Proxy(args.server_uri)

    server_args = jobserver.get_analysis_args('initializing ' + hostname)

    ## perform instantiation
    print('recieved args', args)

    ret = True
    for job in jobserver.get_job('%s %s' % (hostname, ret)):
        print(job)
    #ret = True
    #while True:
    #    job = jobserver.get_job('%s %s' % (hostname, ret))
    #    print(job)
    #    ret =  True #g.batch_process(lsrc, ldst, skipifexist=True, nbtry=3)
