import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import os
import numpy as np
import silence_tensorflow.auto
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
from pydub.utils import make_chunks
import random
from keras.models import load_model
from keras.models import Sequential
import torch
import torchaudio
import pandas as pd
import re
import pathlib2
torch.set_num_threads(1)
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

os.makedirs('SNR_seg',exist_ok=True)

targets = [9,6,3,0,-3,-6,-9] #Change this when u add lower SNRs like: [9,6,3,0,-3,-6,-9,-12,-15,-18,-21,-24,-27,-30]

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('SNR_seg',directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('SNR_seg/noise',directory) 
 os.makedirs(path, exist_ok=True) 


Speech_dir = os.listdir('creating_dataset2/dataset_all4/clean')

Path_to_speech = 'creating_dataset2/dataset_all4/clean'
#Path_to_noise = 'creating_dataset2/dataset_all4/noise/-6dB'

SNR_check = np.zeros((len(Speech_dir),2))
Segment_length_in_seconds = 0.1
Sampling_period = 1 / 16000
N_samples_per_seg = int(Segment_length_in_seconds / Sampling_period)

for target_db in range (0,len(targets)):
  Noise_dir = os.listdir('creating_dataset2/dataset_all4/noise/'+str(targets[target_db])+'dB')
  Path_to_noise = 'creating_dataset2/dataset_all4/noise/'+str(targets[target_db])+'dB'
  Path_to_save = 'SNR_seg/'+str(targets[target_db])+'dB'
  Path_to_save_for_noise = 'SNR_seg/noise/'+str(targets[target_db])+'dB'
  SNR_target = targets[target_db]
  for No_of_data in range (0,len(Speech_dir)):
   #print(No_of_data)
   wav = read_audio(Path_to_speech+'/'+Speech_dir[No_of_data], sampling_rate=16000)
   
   samplerate, Speech_data = wavfile.read("creating_dataset2/dataset_all4/-6dB/"+ Speech_dir[No_of_data])
   Bit_Check = wave.open("creating_dataset2/dataset_all4/-6dB/"+ Speech_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Speech_data = Speech_data/(2**(bit_depth-1))


   samplerate, Noise_data = wavfile.read(Path_to_noise+'/'+Noise_dir[No_of_data])
   Bit_Check = wave.open(Path_to_noise+'/'+ Noise_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Noise_data = Speech_data/(2**(bit_depth-1))


   Noise = read_audio(Path_to_noise+'/'+Noise_dir[No_of_data], sampling_rate = 16000)
   speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
   #speech_timestamps2 = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
   #print(len(speech_timestamps))
   if not(len(speech_timestamps) == 0):
    Noise_timestamps = speech_timestamps
    Overlapping_section_of_noise = collect_chunks(Noise_timestamps,Noise)
    Speech_filtered_by_VAD = collect_chunks(speech_timestamps, wav)
    Speech_Numpy = Speech_filtered_by_VAD.numpy()
    Noise_Numpy = Overlapping_section_of_noise.numpy()
    Power_of_Speech = np.sum(Speech_Numpy ** 2)
    Power_of_Noise = np.sum(Noise_Numpy ** 2)
    #snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)


    Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))
    Noise_Numpy = Multiple * Noise_Numpy
    Power_of_Noise = np.sum(Noise_Numpy ** 2)
    # Print the calculated SNR to verify that it matches the target SNR
    snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)
    #print(snr_global)
    #Adjusted_noisy_speech = wav+Multiple*Noise
    Adjusted_noisy_speech = Speech_data+Multiple*Noise_data
    sf.write(Path_to_save+'/'+str(Speech_dir[No_of_data]), Adjusted_noisy_speech, 16000, 'PCM_16')
    sf.write(Path_to_save_for_noise+'/'+str(Speech_dir[No_of_data]), Multiple*Noise_data, 16000, 'PCM_16')

    #save_audio(Path_to_save+'/'+str(Speech_dir[No_of_data]),
    #         Adjusted_noisy_speech, sampling_rate=16000)
    #save_audio(Path_to_save_for_noise+'/'+str(Speech_dir[No_of_data]),
    #         Multiple*Noise, sampling_rate=16000) 

#     Segments_in_audio = math.floor(len(Speech_Numpy) / N_samples_per_seg)
#     total_sum = 0
#     for segments in range(Segments_in_audio):
#       start_idx = segments * N_samples_per_seg
#       end_idx = start_idx + N_samples_per_seg

#       Current_speech_segment = Speech_Numpy[start_idx:end_idx]
#       Current_noise_segment = Noise_Numpy[start_idx:end_idx]

#       # Calculate the power of speech and noise for the current segment
#       Power_of_Speech = np.sum(Current_speech_segment ** 2)
#       Power_of_Noise = np.sum(Current_noise_segment ** 2)

#       # Calculate the target segmental SNR in dB

#       # Calculate the required value of Multiple to achieve the target SNR for the current segment
#       Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))

#       # Adjust the noise segment to match the target SNR
#       Current_noise_segment *= Multiple

#       # Recalculate the power of noise with the adjusted noise segment
#       Power_of_Noise_adjusted = np.sum(Current_noise_segment ** 2)

#       # Calculate the segmental SNR for the current segment and add it to the total sum
#       total_sum += Power_of_Speech / Power_of_Noise_adjusted
#       #total_sum += Power_of_Speech / Power_of_Noise


#    Pre_db = total_sum / Segments_in_audio

#    Seg_SNR = 10 * np.log10(Pre_db)

  else:
   
   Speech_Numpy = wav.numpy()
   Noise_Numpy = Noise.numpy()
   Power_of_Speech = np.sum(Speech_Numpy ** 2)
   Power_of_Noise = np.sum(Noise_Numpy ** 2)
   #snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)


   Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))
   Noise_Numpy = Multiple * Noise_Numpy
   Power_of_Noise = np.sum(Noise_Numpy ** 2)
    # Print the calculated SNR to verify that it matches the target SNR
   snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)
   Adjusted_noisy_speech = Speech_data+Multiple*Noise_data
   sf.write(Path_to_save+'/'+str(Speech_dir[No_of_data]), Adjusted_noisy_speech, 16000, 'PCM_16')
   sf.write(Path_to_save_for_noise+'/'+str(Speech_dir[No_of_data]), Multiple*Noise_data, 16000, 'PCM_16')
    # Segments_in_audio = math.floor(len(Speech_Numpy) / N_samples_per_seg)
    # total_sum = 0

#    for segments in range(Segments_in_audio):
#     start_idx = segments * N_samples_per_seg
#     end_idx = start_idx + N_samples_per_seg

#     Current_speech_segment = Speech_Numpy[start_idx:end_idx]
#     Current_noise_segment = Noise_Numpy[start_idx:end_idx]

#     # Calculate the power of speech and noise for the current segment
#     Power_of_Speech = np.sum(Current_speech_segment ** 2)
#     Power_of_Noise = np.sum(Current_noise_segment ** 2)

#     # Calculate the target segmental SNR in dB
#     #SNR_target = SNR_target

#     # Calculate the required value of Multiple to achieve the target SNR for the current segment
#     #Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))

#     # Adjust the noise segment to match the target SNR
#     #Current_noise_segment *= Multiple

#     # Recalculate the power of noise with the adjusted noise segment
#     #Power_of_Noise_adjusted = np.sum(Current_noise_segment ** 2)

#     # Calculate the segmental SNR for the current segment and add it to the total sum
#     #total_sum += Power_of_Speech / Power_of_Noise_adjusted
#     total_sum += Power_of_Speech / Power_of_Noise
#  # Calculate the mean of the segmental SNRs
#    Pre_db = total_sum / Segments_in_audio

#  # Calculate the final segmental SNR in dB
#    Seg_SNR = 10 * np.log10(Pre_db)
#    SNR_check[No_of_data,:] = [snr_global,Seg_SNR]
  


# myFile = open('SNR_check.txt', 'r+')
# np.savetxt(myFile, SNR_check)
# myFile.close()





  
pass




# wav = read_audio('test.wav', sampling_rate=16000)
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
# # save_audio('only_speech.wav',
# #            collect_chunks(speech_timestamps, wav), sampling_rate=16000) 
# Noise = read_audio('noise_1005.wav', sampling_rate = 16000)
# Noise_overlap = get_speech_timestamps(wav, model, sampling_rate=16000)
# # print(str(Noise_overlap))
# # Noise_overlap = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", str(Noise_overlap))
# # print(Noise_overlap)
# Noise_overlap = collect_chunks(speech_timestamps,Noise)
# Speech_only = collect_chunks(speech_timestamps, wav)
# # Speech_only = Speech_only.numpy()
# Multiple = 1
# # Noise_overlap = Multiple*Noise_overlap.numpy()
# # #print(np.shape(Speech_only))

# # snr_global = 10*np.log10(np.sum(Speech_only**2)/np.sum(Noise_overlap**2))
# # print(snr_global)
# # target = -6

# Speech_Numpy = Speech_only.numpy()
# Noise_Numpy = Multiple * Noise_overlap.numpy()

# # Calculate the power of speech and noise
# Power_of_Speech = np.sum(Speech_Numpy ** 2)
# Power_of_Noise = np.sum(Noise_Numpy ** 2)

# # Calculate the target SNR in dB
# SNR_target = -30

# # Calculate the required value of Multiple to achieve the target SNR
# Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))
# Noise_Numpy = Multiple * Noise_Numpy
# Power_of_Noise = np.sum(Noise_Numpy ** 2)
# # Print the calculated SNR to verify that it matches the target SNR
# snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)
# print(snr_global)



# #take 100ms samples
# Segment_length_in_seconds = 0.1
# Sampling_period = 1 / 16000
# N_samples_per_seg = int(Segment_length_in_seconds / Sampling_period)

# Segments_in_audio = math.floor(len(Speech_Numpy) / N_samples_per_seg)
# total_sum = 0

# for segments in range(Segments_in_audio):
#     start_idx = segments * N_samples_per_seg
#     end_idx = start_idx + N_samples_per_seg

#     Current_speech_segment = Speech_Numpy[start_idx:end_idx]
#     Current_noise_segment = Noise_Numpy[start_idx:end_idx]

#     # Calculate the power of speech and noise for the current segment
#     Power_of_Speech = np.sum(Current_speech_segment ** 2)
#     Power_of_Noise = np.sum(Current_noise_segment ** 2)

#     # Calculate the target segmental SNR in dB
#     SNR_target = SNR_target

#     # Calculate the required value of Multiple to achieve the target SNR for the current segment
#     Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))

#     # Adjust the noise segment to match the target SNR
#     Current_noise_segment *= Multiple

#     # Recalculate the power of noise with the adjusted noise segment
#     Power_of_Noise_adjusted = np.sum(Current_noise_segment ** 2)

#     # Calculate the segmental SNR for the current segment and add it to the total sum
#     total_sum += Power_of_Speech / Power_of_Noise_adjusted

# # Calculate the mean of the segmental SNRs
# Pre_db = total_sum / Segments_in_audio

# # Calculate the final segmental SNR in dB
# Seg_SNR = 10 * np.log10(Pre_db)
# Speech_Noise = Speech_only+Multiple*Noise_overlap
# print(Seg_SNR)
# save_audio('whatevs.wav',
#             Speech_Noise, sampling_rate=16000) 




#implemenet SileroVAD then do SegmentSNR as Chehresa used it