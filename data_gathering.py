# Imports
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_gaze import gaze_exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--r", help="Number of repetitions to run",default=10, type=int)
args = parser.parse_args()

# Define some variables
board_name = 'muse2'
experiment = 'visual_gaze'
session = 1
subject = 1 
n_classes = 8 # right, up, left, down, resting, blink, rwink, lwink

# Initiate EEG device
eeg_device = EEG(device=board_name)

# Create output filename
save_fn = generate_save_fn(board_name, experiment, subject, session_nb=session)

# Run experiment 
gaze_exp.present(eeg=eeg_device, save_fn=save_fn, n_reps = args.r, n_classes = n_classes)
