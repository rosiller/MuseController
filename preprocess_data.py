import os 
from glob import glob
from utils.plotting import import_files, bandpass
import numpy as np

def re_filter_all_data(data_dir,frequency_band):
    filenames = [os.path.basename(x) for x in glob(os.path.join(data_dir, "*.csv"))]
    trials, labels = import_files(data_dir,filenames)
    trials = bandpass(trials,low_f=frequency_band[0], high_f=frequency_band[1])
    return trials, labels

data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '2-Data/1-RawData/visual_gaze/local/muse2/subject0001/session001/'))

frequency_band=[0.1,5]
trials, labels = re_filter_all_data(data_dir,frequency_band)

# Save data
data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '2-Data/2-PreprocessedData/'))

np.save(os.path.join(data_path,'trials'),trials)
np.save(os.path.join(data_path,'labels'),labels)

print(f'Saved {trials.shape[0]} trials ({trials.shape[1]} channels and {trials.shape[2]} datapoints) \n\
        in {data_path}')