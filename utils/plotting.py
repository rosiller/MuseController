import numpy as np
import torch as t
import matplotlib.pyplot as plt
from os import path

plt.rcParams["figure.figsize"] = (16, 6) # (w, h)


LABEL_DICT_4C = {0:'rest',
                1:'blink',
                2:'rwink',
                3:'lwink'}
LABEL_DICT_5C = {0:'rest',
                1:'blink',
                2:'rwink',
                3:'lwink',
                4:'noise'}
LABEL_DICT_8C = {0:'rwink',
                1:'left',
                2:'right',
                3:'lwink',
                4:'rest',
                5:'up',
                6:'blink',
                7:'down'}

channels=['TP9','AF7','AF8','TP10']


def import_files(datadir, filenames, nb_classes=8):
    datapoints=256
    labels_accum = np.empty((0))
    trial_accum = np.empty((0,len(channels),datapoints))
    
    for filename in filenames:
        filepath = datadir + filename
        filepath = path.join(datadir,filename)
        labels = []

        #columns to import, all but 'Right AUX'
        usecols = np.arange(7)
        usecols = np.delete(usecols,5)

        eeg = np.genfromtxt(filepath, delimiter=',', skip_header = 15, usecols=usecols)
        
        try:
            # scale the time
            eeg[:,0]= eeg[:,0] - eeg[:,0].min()
        except:
            print(f'error with {filename}')
            continue
        
        # prepare the labels 
        for i in range(nb_classes):
            labels.append(eeg[np.where(eeg[:,5]==i+1)[0],0])
        trials, labels = extract_trials_and_labels(labels, eeg)
        
        trial_accum = np.append(trial_accum, trials,axis=0)
        labels_accum = np.append(labels_accum, labels)
    print(f'imported {labels_accum.shape[0]} trials')
    return trial_accum, labels_accum

def import_file(filename, nb_classes=8):
    #columns to import, all but 'Right AUX'
    usecols = np.arange(7)
    usecols = np.delete(usecols,5)
    
    # import the csv remove first 15 rows which have empty marker column
    eeg = np.genfromtxt(filename, delimiter=',', skip_header = 15, usecols=usecols)
    
    # scale the time
    eeg[:,0]= eeg[:,0] - eeg[:,0].min()
    
    # prepare the labels 
    labels = []

    for i in range(nb_classes):
        labels.append(eeg[np.where(eeg[:,5]==i+1)[0],0])
    return eeg, labels

def mark_area(labels, color):
    for ind, line in enumerate(labels):
        if ind ==0:
            plt.axvspan(line,line+1,alpha=0.5,color=color)
        else:
            plt.axvspan(line,line+1,alpha=0.5,color=color,label='_nolegend_')

def get_label_string_or_index(label,nb_classes=8):
    
    # Pick appropriate dictionary
    if nb_classes ==4:
        label_dict = LABEL_DICT_4C
    elif nb_classes == 5:
        label_dict = LABEL_DICT_5C
    else:
        label_dict = LABEL_DICT_8C
    
    # Return either string or index
    if type(label)==str:
        return [k for k,v in label_dict.items() if v==label][0]
    elif type(label)==int:
        return label_dict[label]
    else:
        return 1/0
        
def plot_eeg(eeg, labels, channel=3, start=0, end=100):
    plt.plot(eeg[start:end,0],eeg[start:end,channel])
    plt.xlabel('Time [s]',size=15)
    plt.ylabel('Magnitude [$\mu$V]',size=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.show
    if end > eeg.shape[0]-1 | (end == None ) | (end == False):
        end = eeg.shape[0]-1
    plt.xlim(eeg[start,0],eeg[end,0])
    
    mark_area(labels[0],color='y') #rwink
    mark_area(labels[1],color='r') # left
    mark_area(labels[2],color='g') # Right
    mark_area(labels[3],color='k') # Lwink
    mark_area(labels[4],color='cyan') # Rest
    mark_area(labels[5],color='brown') # Up
    mark_area(labels[6],color='orange') # blink
    mark_area(labels[7],color='magenta') # down
    
    label_list = ['EEG']
    for label in range(len(labels)):
        label_list.append(get_label_string(int(label)))
    plt.legend(label_list)
    
def get_sample_from_label(labels, eeg, color='r'):
    datapoints = 256
    trial_data = np.empty((len(labels),4,datapoints))
    trial_labels = labels.shape
    for ind, label in enumerate(labels):
        start = np.where(eeg[:,0]>=label)[0][0]
        end = np.where(eeg[:,0]<=label+1)[0][-1]
        if end-start < datapoints:
            end += datapoints-(end-start)
        elif end-start>datapoints:
            end-= (end-start)-datapoints
        for channel in range(4):
            trial_data[ind,channel] = eeg[start:end,channel+1]
    return trial_data, trial_labels

def extract_trials_and_labels(markers, eeg, nb_classes=8):
    datapoints = 256
    all_trials = np.empty((0,4,datapoints))    
    labels = np.empty(0)
    for i in range(nb_classes):
        trials, trial_labels =  get_sample_from_label(markers[i], eeg)
        all_trials = np.append(all_trials, trials, axis=0)
        labels = np.append(labels,np.repeat(i,trial_labels))
    return all_trials, labels

def bandpass(input_, low_f, high_f, device=t.device('cuda')):
    """
    Parameters:
     - input:                   numpy
     - low_f:                   float, lowest frequency which will be allowed                         
     - high_f:                  float, highest frequency which will be allowed
    
    Returns: filtered input tensor

    """
    input_ = t.from_numpy(input_)
    pass1 = t.abs(t.fft.rfftfreq(input_.shape[-1],1/160)) > low_f
    pass2 = t.abs(t.fft.rfftfreq(input_.shape[-1],1/160)) < high_f
    fft_input = t.fft.rfft(input_)
    return t.fft.irfft(fft_input.to(device) * pass1.to(device) * pass2.to(device)).cpu().numpy()

def plot_individual_label(data, labels, label, channel, slice_to_plot=[]):
    fs = data.shape[-1]
    if not slice_to_plot:
        trials=data.shape[0]
        for i in range(trials):
            if labels[i]==label:
                plt.plot(np.arange(0,1,1/fs),data[i,channel])
    else:
        for ind, trial in enumerate(labels[slice_to_plot]):
            if trial==label:
                ind = slice_to_plot.start+ind*slice_to_plot.step
                plt.plot(np.arange(0,1,1/fs),data[ind,channel])
    plt.legend([channels[channel]])
    plt.xlabel('Time [s]')
    plt.ylabel('EEG signal [mV]')
    plt.show()