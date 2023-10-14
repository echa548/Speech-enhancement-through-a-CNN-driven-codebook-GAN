import numpy as np
import os
import wave
import scipy.signal as sps
from pydub import AudioSegment
import wave
from pydub.utils import make_chunks
import random

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

#This file will make the necessary training data for the Noise GAN.
os.chdir('../..')

N_fft = 1024
def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')
 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft


Data_number_Mixture =0
Data_number_clean = 0
chunk_length_ms = 32
samplerate = 16000

Folder_path = 'Dataset/VAD_SNR'
Folder_path_save = 'Dataset/VAD_SNR/'
Folder_path_to_access = 'Dataset/VAD_SNR/'
Path_Before_VAD = 'Dataset/dataset'

dummy_directories = os.listdir('Dataset/VAD_SNR/3dB')

targets = [9,6,3,0,-3,-6,-9]


Mixture_list = []
Clean_list = []
for target_db in range (0,len(targets)):
 Path_of_noisy_mixture = Folder_path_to_access +str(targets[target_db])+'dB'+'/'
 Path_of_real_noise_files = Folder_path_to_access +'noise/'+str(targets[target_db])+'dB'+'/'
 Noise_files = os.listdir(Folder_path_to_access +'noise/'+str(targets[target_db])+'dB')
 Noisy_mixture_files = os.listdir(Folder_path_to_access + str(targets[target_db])+'dB')

 for No_of_data in range (0,len(Noisy_mixture_files)):

   myaudio = AudioSegment.from_file(Path_of_noisy_mixture + Noisy_mixture_files[No_of_data], "wav")
   real_noise = AudioSegment.from_file(Path_of_real_noise_files + Noise_files[No_of_data], "wav")
    
    
   chunks = make_chunks(myaudio, chunk_length_ms)
   chunks2 = make_chunks(real_noise, chunk_length_ms)
    
   shuffled_indices = list(range(len(chunks)))
   random.shuffle(shuffled_indices)
   num_chunks_to_select = len(chunks) // 10
   selected_indices = shuffled_indices[:num_chunks_to_select]

   for indices in selected_indices:
        Mixture_chunk = chunks[indices]
        Mixture_data = np.array(Mixture_chunk.get_array_of_samples())
        Mixture_data = np.transpose(Mixture_data)
        Bit_Check = wave.open(Path_of_noisy_mixture+ Noisy_mixture_files[No_of_data], 'rb')
        bit_depth = Bit_Check.getsampwidth() * 8


        Mixture_data = Mixture_data/(2**(bit_depth-1))
        Rectangular_windowed_signal = Mixture_data
        FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
        Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
        PSD_window_scaling = np.sum(Hann_window**2)
        Mixture_list.append((np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling))
        

        Real_noise_chunk = chunks2[indices]
        Real_noise_chunk = np.array(Real_noise_chunk.get_array_of_samples())
        Real_noise_chunk = np.transpose(Real_noise_chunk)
        Bit_Check = wave.open(Path_of_real_noise_files+ Noise_files[No_of_data], 'rb')
        bit_depth = Bit_Check.getsampwidth() * 8
        Real_noise_chunk = Real_noise_chunk/(2**(bit_depth-1))
        Rectangular_windowed_signal = Real_noise_chunk
        FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
        Clean_list.append((np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling))
        Mixture_chunk = 0
        Real_noise_chunk = 0
        

Mixture_PSD = np.zeros((len(Mixture_list),1024))
Clean_PSD = np.zeros((len(Clean_list),1024))

for data_point in range (0, len(Mixture_list)):
 Mixture_PSD[data_point,:] = Mixture_list.pop()
 Clean_PSD[data_point,:] = Clean_list.pop()
 
np.save('Compendium/Models-Setup/GAN-Setup/Noisy_Mixture_PSDs',Mixture_PSD)

np.save('Compendium/Models-Setup/GAN-Setup/Pure_Noise_PSDs',Clean_PSD)