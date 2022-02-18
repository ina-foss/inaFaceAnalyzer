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
import sys
import os
import socket
from collections import namedtuple
import pandas as pd
import inaFaceAnalyzer.commandline_utils as ifacu
from inaFaceAnalyzer.display_utils import ass_subtitle_export, video_export


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

    # parse command line arguments
    parser = ifacu.new_parser(description)
    parser.add_argument(dest='server_uri', help=hserver_uri)
    ifacu.add_batchsize(parser)
    args = parser.parse_args()

    # setup network configuration
    hostname = socket.gethostname()
    jobserver = Pyro4.Proxy(args.server_uri)

    # get processing settings from server, and update with command line arguments
    server_args = jobserver.get_analysis_args('initializing ' + hostname)
    print('recieved args', server_args)
    server_args['batch_size'] = args.batch_size
    nt = namedtuple('inaFaceAnalyzerArgs', server_args)
    args = nt(**server_args)

    # to be used at engine inference
    dargs = {}
    if args.fps:
        dargs = {'fps': args.fps}

    ## perform engine instantiation
    engine = ifacu.engine_factory(args)


    ret = 'first call'
    stopit = False
    while not stopit:
        try:
            src, dst, dst_ass, dst_mp4 = jobserver.get_job('%s %s' % (hostname, ret))
        except StopIteration:
            print('all jobs are done')
            stopit = True
            sys.exit(0)

        print('received job', src, dst, dst_ass, dst_mp4)
        df = None

        ret = ''
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            df = engine(src, **dargs)
            df.to_csv(dst, index=False)
            ret += '%s done ' % dst

        if dst_ass == dst_ass and not os.path.exists(dst_ass):
            os.makedirs(os.path.dirname(dst_ass), exist_ok=True)
            if df is None:
                df = pd.read_csv(dst)
            ass_subtitle_export(src, df, dst_ass, analysis_fps=args.fps)
            ret += '%s done ' % dst_ass

        if dst_mp4 == dst_mp4 and not os.path.exists(dst_mp4):
            os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
            if df is None:
                df = pd.read_csv(dst)
            video_export(src, df, dst_mp4, analysis_fps=args.fps)
            ret += '%s done ' % dst_mp4