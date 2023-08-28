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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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


Mixture_list = []
Clean_list = []
for target_db in range (0,len(targets)):
 Path_of_noisy_mixture = 'SNR_seg/'+str(targets[target_db])+'dB'+'/'
 Path_of_real_speech_files = 'SNR_seg/clean/'
 real_speech_files = os.listdir('SNR_seg/clean')
 Noisy_mixture_files = os.listdir('SNR_seg/'+str(targets[target_db])+'dB')
 print(targets[target_db])


 for No_of_data in range (0,len(Noisy_mixture_files)):

   myaudio = AudioSegment.from_file(Path_of_noisy_mixture + Noisy_mixture_files[No_of_data], "wav")
   real_speech = AudioSegment.from_file(Path_of_real_speech_files + real_speech_files[No_of_data], "wav")
    
    
   chunks = make_chunks(myaudio, chunk_length_ms)
   chunks2 = make_chunks(real_speech, chunk_length_ms)
    
   shuffled_indices = list(range(len(chunks)))
   random.shuffle(shuffled_indices)
   num_chunks_to_select = len(chunks) // 10
   selected_indices = shuffled_indices[:num_chunks_to_select]

   for indices in selected_indices:
        Mixture_chunk = chunks[indices]
        Mixture_data = np.array(Mixture_chunk.get_array_of_samples())
        #print(np.shape(Mixture_data))
        Mixture_data = np.transpose(Mixture_data)
        Bit_Check = wave.open(Path_of_noisy_mixture+ Noisy_mixture_files[No_of_data], 'rb')
        bit_depth = Bit_Check.getsampwidth() * 8


        Mixture_data = Mixture_data/(2**(bit_depth-1))
        Rectangular_windowed_signal = Mixture_data
        FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
        Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
        PSD_window_scaling = np.sum(Hann_window**2)
        Mixture_list.append((np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling))
        


        Real_speech_chunk = chunks2[indices]
        Real_speech_chunk = np.array(Real_speech_chunk.get_array_of_samples())
        #print(np.shape(Real_noise_chunk))
        Real_speech_chunk = np.transpose(Real_speech_chunk)
        Bit_Check = wave.open(Path_of_real_speech_files+ real_speech_files[No_of_data], 'rb')
        bit_depth = Bit_Check.getsampwidth() * 8
        Real_speech_chunk = Real_speech_chunk/(2**(bit_depth-1))
        Rectangular_windowed_signal = Real_speech_chunk
        FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
        Clean_list.append((np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling))
        Mixture_chunk = 0
        Real_speech_chunk = 0
        
Mixture_PSD = np.zeros((len(Mixture_list),1024))
Clean_PSD = np.zeros((len(Clean_list),1024))

for data_point in range (0, len(Mixture_list)):
 Mixture_PSD[data_point,:] = Mixture_list[data_point]
 Clean_PSD[data_point,:] = Clean_list[data_point]
 




np.save('Mixture_PSD',Mixture_PSD)


np.save('Clean_PSD',Clean_PSD)