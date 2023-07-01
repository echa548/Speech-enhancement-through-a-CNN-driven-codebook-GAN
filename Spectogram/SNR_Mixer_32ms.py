from scipy.io import wavfile
import math
import numpy as np
from scipy import signal
from pathlib import Path
import scipy.signal as sps
from scipy.signal import butter, lfilter
import soundfile as sf
import matplotlib.pyplot as plt
import pydub
import uuid
import os
from pydub import AudioSegment, effects
import wave

os.chdir('C:/Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700')
directories = os.listdir('Spectogram/MY_Experimenting_Folder/Processed_audio/processed_noise')
rate = 16000
Sampling_interval = 1/rate
for No_of_data in range (0,len(directories)):
   samplerate, data = wavfile.read("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_noise/"+ directories[No_of_data])
   #print(samplerate)
   Bit_Check = wave.open("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_noise/"+ directories[No_of_data], 'rb')
   Overlaps = math.floor(len(data)/1024)
   for No_of_overlaps in range (0,Overlaps-1):
     Truncated_signal = data[0+1024*No_of_overlaps:1024+1024*No_of_overlaps]
     print(len(Truncated_signal)/rate)
     #padded_signal = np.pad(Truncated_signal,(0,512), 'constant')
     f, t, Lxx = signal.spectrogram(x = Truncated_signal, fs = samplerate, window = 'hann',nperseg = 512,noverlap = 384,nfft = 1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
     #print(t)
     plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
     plt.colorbar(label='Decibels')
     plt.ylabel('Frequency [Hz]')
     plt.xlabel('Time [sec]')
     plt.savefig('Spectogram/Spectrogram_Plots/Mag/' + str(No_of_overlaps) + 'Magnitude_Plot_Mono.png')
     plt.close()