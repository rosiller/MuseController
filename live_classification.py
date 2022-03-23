# Imports
from eegnb.devices.eeg import EEG
import torch as t
import time
import os
from utils.plotting import bandpass, get_label_string_or_index
import pyautogui
import argparse
import numpy as np
from utils.data_utils import get_stream_and_inlet
from muselsl import constants as mlsl_cnsts

parser = argparse.ArgumentParser()
parser.add_argument("--c", help="Number of classifications to run",nargs='?', const=10, default=10, type=int)
args = parser.parse_args()
  
# Define some variables
board_name = 'muse2'
nb_classes = 5
# Initiate EEG device
eeg_device = EEG(device=board_name)

data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '1-Model'))

model = t.load(os.path.join(data_path,f'GazeModel_{nb_classes}class.net'))
device = t.device("cuda:0")
model.eval().to(device)  
eeg_device.start(fn="None")
window = 1
n_samples = int(window*256)
accum = np.zeros((n_samples, 4))
inlet_eeg = get_stream_and_inlet(sensor='EEG')
k=0

for i in range(args.c):
    
    samples_eeg = np.array(inlet_eeg.pull_chunk(timeout=1.0, max_samples=mlsl_cnsts.LSL_EEG_CHUNK)[0])
    if len(samples_eeg)!=0 & eeg_device.stream_process.is_alive():
        #Plot
        samples_eeg = samples_eeg[:,:-1]
        accum = np.vstack([accum, samples_eeg])

        accum = accum[-n_samples:]
        k+=1
        if k==4:
            sample_t = bandpass(accum.T,low_f=0.1, high_f=5)

            sample_t= t.tensor(sample_t) 
            output = model(sample_t.to(device).unsqueeze(0).unsqueeze(0).float()).argmax().item()
            output = get_label_string_or_index(output,nb_classes=nb_classes)
            print(f'{output}')

            k=0

            if output =='blink':
                pyautogui.hotkey('ctrl','alt','p')    
            elif output == 'rwink':
                # pyautogui.press('space')
                pyautogui.hotkey('ctrl','alt','u')
            elif output == 'lwink':
                pyautogui.hotkey('ctrl','alt','b')
                # pyautogui.press('space')
    else:
        print('resting',eeg_device.stream_process.is_alive())
        time.sleep(0.2)

    


if eeg_device.stream_process.is_alive():
        print('Closing connection')
        eeg_device.stop()
        eeg_device.stream_process.terminate()