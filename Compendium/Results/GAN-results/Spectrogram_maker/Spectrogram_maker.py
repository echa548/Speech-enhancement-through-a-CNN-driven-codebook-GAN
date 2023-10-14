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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

directories = os.listdir('Audio')

#This file was used to generate the spectrograms provided in the results.

def Generate_Mono_Mag_Phase_Spectrogram (data,fs,dir):
     f, t, Lxx = signal.spectrogram(x = data, fs = fs, window = 'hann',nperseg = 512,noverlap = 384,nfft = 1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
     plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
     plt.colorbar(label='Decibels')
     plt.ylabel('Frequency [Hz]')
     plt.xlabel('Time [sec]')
     plt.savefig('Spectrograms/'+str(dir)+'Magnitude_Plot_Mono.png')
     plt.close()
     
def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')
 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft
     
for i in range (0,len(directories)):
          samplerate, data = wavfile.read('Audio/'+directories[i])
          Bit_Check = wave.open('Audio/'+directories[i],'rb')
          bit_depth = Bit_Check.getsampwidth() * 8
          data = data/(2**(bit_depth-1))
          Generate_Mono_Mag_Phase_Spectrogram (data,samplerate,directories[i])
          pass

