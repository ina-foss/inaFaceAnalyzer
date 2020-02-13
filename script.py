import argparse
import glob
import os
import warnings
import progressbar

parser = argparse.ArgumentParser(description='Detect and classify faces by gender from a video. Store segmentations into CSV files')
parser.add_argument('-i', '--input', nargs='+', help='Input media to analyse. May be a full path to a media (/home/david/test.mp4), a list of full paths (/home/david/test.mp4 /tmp/mymedia.avi), or a regex input pattern ("/home/david/myaudiobooks/*.mp4")', required=True)
parser.add_argument('-o', '--output_directory', help='Directory used to store classification info. Resulting files have same base name as the corresponding input media, with csv extension. Ex: mymedia.MP4 will result in mymedia.csv', required=True)


parser.add_argument('-s', '--time_offset', help = 'time in milliseconds from which we begin extraction of the frames in video', required=False)
parser.add_argument('-f', '--nframes', help = 'process every n frames', required = False)
parser.add_argument('-t', '--mode', help = 'With or without tracking mode', choices = ['on','off'], required = False)

parser.add_argument('-k', '--ktracking', help = 'Used in case of tracking: re-detect faces every k frames', required= False)


args = parser.parse_args()
input_files = []
for e in args.input:
    input_files += glob.glob(e)
assert len(input_files) > 0, 'No existing media selected for analysis! Bad values provided to -i (%s)' % args.input

odir = args.output_directory
assert os.access(odir, os.W_OK), 'Directory %s is not writable!' % odir


if args.mode == 'on' and args.ktracking is None :
    parser.error("--mode requires --ktracking ! ")
    
    
n_frames = args.nframes
if n_frames:
    n_frames = int(n_frames)
else: 
    n_frames S= 1
offset = args.time_offset
if offset:
    offset = int(offset)

track_mode = args.mode
track_frames = int(args.ktracking)

from inaFaceGender import GenderVideo, info2csv

gen = GenderVideo()

bar = progressbar.ProgressBar(maxval=10000, \
widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    bar.start()
    if track_mode == 'on':       
        for i, e in enumerate(input_files):
            #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
            base, _ = os.path.splitext(os.path.basename(e))
            info2csv(gen.detect_with_tracking(e, track_frames, n_frames, offset ), '%s/%s_tracked.csv' % (odir, base))
            bar.update(i)
        bar.finish()
    else:       
        for i, e in enumerate(input_files):
            #print('\n processing file %d/%d: %s' % (i+1, len(input_files), e))
            base, _ = os.path.splitext(os.path.basename(e))
            info2csv(gen(e, n_frames, offset), '%s/%s.csv' % (odir, base))
            bar.update(i)
        bar.finish()
