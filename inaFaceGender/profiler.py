import cProfile
import pstats
import io
import re
import pandas as pd

class IfaProfile(cProfile.Profile):
    def __init__(self, filters = None, **kwargs):
        super().__init__(**kwargs)
        if filters is None:
            filters = ['__call__|video_iterator|preprocess_face',
                       'inaFaceGender2/inaFaceGender']
        self.filters = filters
    def stats2df(self):
        stream = io.StringIO()
        s = pstats.Stats(pr, stream=stream)
        
        #filter here
        s.print_stats(*self.filters)

        lines = [re.sub(' +', ' ', e.strip()) for e in stream.getvalue().split('\n')[5:] if e.strip() != '']

        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=' ')
        
        df['short'] = df['filename:lineno(function)'].map(lambda x: os.path.basename(x))
        df['cum_p'] = df['cumtime'].map(lambda x: x / df.cumtime.max())
        return (df)

with IfaProfile() as pr:
    # do something
    pass
 pr.stats2df()

 
