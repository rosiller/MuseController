import os
from time import time, sleep
from glob import glob
from random import shuffle
from optparse import OptionParser

import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event

from eegnb import generate_save_fn
from eegnb.stimuli import ARROWS
from tqdm import tqdm


__title__ = "Gaze movement"

def present(eeg=None, save_fn=None, n_reps=1, n_classes = 8):
    iti = 2 #Inter trial interval 
    soa = 1  # Stimulus onset asynchrony
    jitter = 0
    n_trials = n_reps * n_classes

    duration = n_trials*soa+(n_trials)*iti+5 # adding 5 seconds to let the signal settle
    record_duration = np.float32(duration)
    markernames = list(range(n_classes))
    
    # Setup trial list
    image_type = markernames * n_reps
    shuffle(image_type)
    trials = DataFrame(dict(image_type=image_type, timestamp=np.zeros(n_trials)))

    def load_image(fn):
        return visual.ImageStim(win=mywin, image=fn)

    # start the EEG stream, will delay 5 seconds to let signal settle

    # Setup graphics
    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg", fullscr=True)
    
    # arrows = list(map(load_image, glob(os.path.join(ARROWS, "*.jpg"))))
    names = [os.path.splitext(os.path.basename(x))[0] for x in glob(os.path.join(ARROWS, "*.jpg"))]
    # print(names)

    # Show the instructions screen
    show_instructions(duration)

    if eeg:
        if save_fn is None:  # If no save_fn passed, generate a new unnamed save file
            save_fn = generate_save_fn(eeg.device_name, "gaze", "unnamed")
            print(
                f"No path for a save file was passed to the experiment. Saving data to {save_fn}"
            )
        try:
            eeg.start(save_fn, duration=record_duration + 5)
        except IndexError:
            print('No Muse was found')
            return 
        disconnected = False

    # Start EEG Stream, wait for signal to settle, and then pull timestamp for start point
    start = time()

    # Iterate through the events
    with tqdm(desc=f'Data gathering', total=n_trials, ascii=True) as bar:

        for ii, trial in trials.iterrows():
            # Inter trial interval
            core.wait(iti + np.random.rand() * jitter)

            # Select and display image
            label = trials["image_type"].iloc[ii]
            image = load_image(glob(os.path.join(ARROWS, "*.jpg"))[label])
            print(glob(os.path.join(ARROWS, "*.jpg")))
            print(glob(os.path.join(ARROWS, "*.jpg"))[label])
            # image = choice(arrows)
            image.draw()

            # Push sample
            if eeg:
                timestamp = time()
                if eeg.backend == "muselsl":
                    marker = [markernames[label]+1]
                    # print(f'marker {marker} {names[label]}')
                else:
                    marker = markernames[label]+1
                eeg.push_sample(marker=marker, timestamp=timestamp)

            mywin.flip()
        
            if not eeg.stream_process.is_alive():
                disconnected = True
                eeg.recording.terminate()
                eeg.stream_process.terminate()
                break

            # offset
            core.wait(soa)
            mywin.flip()
            if len(event.getKeys()) > 10 or (time() - start) > record_duration:
                break

            event.clearEvents()
            bar.update()

    

    show_end_screen(start, duration, disconnected)
    mywin.close()
    
    # Cleanup
    if eeg.stream_process.is_alive():
        sleep(8) # Force wait until file is saved
        print('Terminating it manually')
        eeg.stop()
        eeg.stream_process.terminate()


def show_instructions(duration):

    instruction_text = """
    Welcome to the Gaze experiment! 
 
    Stay still and focus on the centre of the screen. 

    This block will run for %s seconds.

    Press spacebar to continue. 
    
    """
    instruction_text = instruction_text % duration

    # graphics
    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg", fullscr=True)

    mywin.mouseVisible = False

    # Instructions
    text = visual.TextStim(win=mywin, text=instruction_text, color=[-1, -1, -1])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True
    mywin.close()


def show_end_screen(start_time, duration, disconnected=False):
    reason = ''
    
    if disconnected:
        reason = f'Muse disconnected mid-experiment'

    instruction_text = f"""
    Experiment is over

    Please wait until the results file is saved, then press spacebar to finish
    
    {reason}
     
    """

    # graphics
    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg", fullscr=True)

    mywin.mouseVisible = False

    # Instructions
    text = visual.TextStim(win=mywin, text=instruction_text, color=[-1, -1, -1])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True
    mywin.close()
