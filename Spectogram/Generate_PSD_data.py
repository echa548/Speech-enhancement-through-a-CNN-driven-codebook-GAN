import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import silence_tensorflow.auto
from tqdm import tqdm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import wave
from scipy.io import wavfile
import math
from scipy import signal
from pathlib import Path
import scipy.signal as sps
from scipy.signal import butter, lfilter
import soundfile as sf
import pydub
import uuid
from pydub import AudioSegment, effects
import wave
from pydub.utils import make_chunks
import random

os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')

N_fft = 1024
def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')
 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft


#directories = os.listdir('creating_dataset2/dataset_all4/-6dB')
#directories2 = os.listdir('creating_dataset2/dataset_all4/noise/-6dB')
#Mixture_PSD = np.zeros((len(directories),1024))
#Clean_PSD = np.zeros((len(directories),1024))
Data_number_Mixture =0
Data_number_clean = 0
chunk_length_ms = 32
samplerate = 16000

dummy_directories = os.listdir('SNR_seg/3dB')

targets = [9,6,3,0,-3,-6,-9]
Mixture_PSD = np.zeros((len(dummy_directories)*len(targets),1024))
Clean_PSD = np.zeros((len(dummy_directories)*len(targets),1024))

for target_db in range (0,len(targets)):
 Path_of_noisy_mixture = 'SNR_seg/'+str(targets[target_db])+'dB'+'/'
 Path_of_real_noise_files = 'SNR_seg/noise/'+str(targets[target_db])+'dB'+'/'
 Noise_files = os.listdir('SNR_seg/noise/'+str(targets[target_db])+'dB')
 Noisy_mixture_files = os.listdir('SNR_seg/'+str(targets[target_db])+'dB')



 for No_of_data in range (0,len(Noisy_mixture_files)):

   myaudio = AudioSegment.from_file(Path_of_noisy_mixture+ Noisy_mixture_files[No_of_data], "wav") 
   chunks = make_chunks(myaudio, chunk_length_ms)
   chosen_chunk = random.randint(0,len(chunks)-1)
   data = np.array(chunks[chosen_chunk].get_array_of_samples())
   data = np.transpose(data)
   #print(np.shape(data))
   Bit_Check = wave.open(Path_of_noisy_mixture+ Noisy_mixture_files[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
   Rectangular_windowed_signal = data
   FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
   Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
   PSD_window_scaling = np.sum(Hann_window**2)
   Mixture_PSD[No_of_data+(len(Noisy_mixture_files)*target_db),:] = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling) 
   myaudio=0
   chunks = 0
   myaudio = AudioSegment.from_file(Path_of_real_noise_files+ Noise_files[No_of_data], "wav") 
   chunks = make_chunks(myaudio, chunk_length_ms)
   data = np.array(chunks[chosen_chunk].get_array_of_samples())
   #print(data)
   data = np.transpose(data)
   #print(np.shape(data))
   Bit_Check = wave.open(Path_of_real_noise_files+ Noise_files[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
   Rectangular_windowed_signal = data
   FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
   Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
   PSD_window_scaling = np.sum(Hann_window**2)
   Clean_PSD[No_of_data+(len(Noisy_mixture_files)*target_db),:] = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)
   myaudio=0
   chunks = 0
  









#print(np.max(Mixture_PSD))
#print(np.max(Clean_PSD))

np.save('Mixture_PSD',Mixture_PSD)


np.save('Clean_PSD',Clean_PSD)