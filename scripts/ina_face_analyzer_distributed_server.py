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
import pandas as pd
import inaFaceAnalyzer.commandline_utils as ifacu
from inaFaceAnalyzer.face_detector import facedetection_cmdlineparser
import warnings

from collections import namedtuple


@Pyro4.expose
class JobServer(object):
    def __init__(self, args):
        '''
        Create an instance of JobServer, in charge of distributing analyses to
        be performed to a set of remote workers.

        Parameters
        ----------
        args : argument parser results
            see ina_face_analyzer_distributed_server.py -h for a detailed listing
        '''


        # open csv and check mandatory columns are provided
        csv = args.joblist_csv
        print('parsing joblist %s' % csv)
        df = pd.read_csv(csv)
        # check mandatory columns are provided
        assert 'source_path' in df.columns, '%s must contain a source_path column' % csv
        assert 'dest_csv' in df.columns, '%s must contain a dest_csv column' % csv
        for k in df.columns:
            if k not in ['source_path', 'dest_csv', 'dest_ass', 'dest_mp4']:
                warnings.warn('column %s in %s is not supported and will be ignored' % (k, csv))
            df[k] = df[k].map(lambda x: x.strip())
        # shuffle records and drop duplicates
        df = df.drop_duplicates().sample(frac=1).reset_index(drop=True)

        print('setting jobs')
        print('random job record example:', next(df.itertuples()))
        print('Total number of files to process:', len(df))
        self.jobiterator = enumerate(df.itertuples())
        nt = namedtuple('serverargs', args.__dict__)
        self.args = nt(*args.__dict__)
        #self.args = args

    def get_analysis_args(self, msg):
        print(msg)
        return self.args

    def get_job(self, msg):
        #try:
        ijob, job = next(self.jobiterator)
        print('job %d: %s' % (ijob, msg))
        print(job)
        return job
        #except StopIteration:
        #    print('no more jogs')
        #    return None

    # to be implemented - only usefull for image collections
    # def get_njobs(self, msg, nbjobs=20):
    #     print('jobs %d-%d: %s' % (self.i, self.i + nbjobs, msg))
    #     ret = (self.lsource[:nbjobs], self.ldest[:nbjobs])
    #     if len(ret[0]) == 0:
    #         print('All jobs dispatched')
    #     self.lsource = self.lsource[nbjobs:]
    #     self.ldest = self.ldest[nbjobs:]
    #     self.i += nbjobs
    #     return ret

description = '''Server in charge of distributing a list of documents to analyze to
workers distributed over the network. Workers need to have access to a
centralized storage directory for writing output results (mounted with
NFS, sshfs, ...).
To be used jointly with ina_face_analyzer_distributed_worker.py.'''

if __name__ == '__main__':



    parser = ifacu.new_parser(description)

    ## Required arguments
    #ra = parser.add_argument_group('required arguments')

    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")

    h = '''host_address is used by workers to communicate remotely with
    the server. It can be either server's IP adress, or host full name ex: mymachine.my-domain.fr
    '''
    parser.add_argument(dest='host_address', help = h)

    h = ''' joblist_csv is the full path to a csv file storing one line per file
    to process. It may contain up to 4 columns, separated by coma ",".
    "source_path" column (mandatory) contains the path or the url to a file to be processed.
    "dest_csv" (mandatory) contains the path used to write the resulting csv.
    "dest_ass" (non mantatory) is a path used to export results to ASS rich subtitles (displayed in VLC or ELAN).
    "dest_mp4" (non mandatory) is a path used to export results to MP4 video
    with incrusted face bouding boxes and classification results.
    '''
    parser.add_argument(help = h, dest='joblist_csv')

    # face detection
    facedetection_cmdlineparser(parser)

    #### OPTION SKIP IF EXIST

    ifacu.add_framerate(parser)
#    ifacu.add_keyframes(parser)

    # parse command line arguments
    args = parser.parse_args()

    # full name of the host to be used by remote clients
    Pyro4.config.HOST = args.host_address
    daemon = Pyro4.Daemon()

    uri = daemon.register(JobServer(args))
    print("Provide the following objet URI to remove ina_face_analyzer_distributed_workers: ", uri)
    daemon.requestLoop()
