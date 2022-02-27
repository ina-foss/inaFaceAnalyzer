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
# THE SOFTWARE.


import Pyro4
import os
import socket
from collections import namedtuple
import traceback
import time
import inaFaceAnalyzer.commandline_utils as ifacu
from inaFaceAnalyzer.display_utils import ass_subtitle_export, video_export


class IfaWorker:
    def __init__(self, server_uri, batch_size, name_suffix=''):
        # setup network configuration
        self.hostname = hostname = (socket.gethostname() + name_suffix)
        self.jobserver = jobserver = Pyro4.Proxy(server_uri)

        # get processing settings from server, and update with command line arguments
        server_args = jobserver.get_analysis_args('initializing ' + hostname)
        print('recieved args', server_args)
        server_args['batch_size'] = batch_size
        nt = namedtuple('inaFaceAnalyzerArgs', server_args)
        args = nt(**server_args)

        # to be used at engine inference
        self.fps = args.fps

        ## perform engine instantiation
        self.engine = ifacu.engine_factory(args)
        self.msg = 'first call'

    def __call__(self):
        # ask for a job
        # return False if no more jobs are available, else True
        try:
            msg = '%s: %s' % (self.hostname, self.msg)
            src, dst_csv, dst_ass, dst_mp4 = self.jobserver.get_job(msg)
        except StopIteration:
            print('all jobs are done')
            return False

        # init process
        print('received job', src, dst_csv, dst_ass, dst_mp4)
        self.msg = ''
        df = None

        # taskid values: 0 for main analysis, 1 for ass export, 2 for mp4 export
        for taskid, dst in enumerate([dst_csv, dst_ass, dst_mp4]):

            # test if destination path is provided: not None and not Nan
            if not (dst and (dst == dst)):
                continue

            # skip if file already exists
            if os.path.exists(dst):
                self.msg += '%s already exists' % dst
                continue

            # do the job
            b = time.time()
            try:
                # create output directory if needed
                os.makedirs(os.path.dirname(dst), exist_ok=True)

                # task specific action
                if taskid == 0:
                    df = self.engine(src, fps = self.fps)
                    df.to_csv(dst, index=False, float_format='%.2f')
                else:
                    if df is None:
                        df = dst_csv
                    if taskid == 1:
                        print(src, dst, self.fps, df)
                        ass_subtitle_export(src, df, dst, analysis_fps = self.fps)
                    elif taskid == 2:
                        video_export(src, df, dst, analysis_fps = self.fps)
                    else:
                        raise NotImplementedError()

                self.msg += '%s done in %.1f sec' % (dst, time.time() - b)
            except BaseException:
                self.msg += 'problem with %s\n' % dst
                self.msg += traceback.format_exc()

        return True


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

    # create worker instance
    worker = IfaWorker(args.server_uri, args.batch_size)

    # iterate while jobs are available
    while worker():
        pass
