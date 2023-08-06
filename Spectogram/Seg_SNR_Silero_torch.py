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
   
   samplerate, Speech_data = wavfile.read(Path_to_speech+"/"+ Speech_dir[No_of_data])
   Bit_Check = wave.open(Path_to_speech+"/"+ Speech_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Speech_data = Speech_data/(2**(bit_depth-1))


   samplerate, Noise_data = wavfile.read(Path_to_noise+'/'+Noise_dir[No_of_data])
   Bit_Check = wave.open(Path_to_noise+'/'+ Noise_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Noise_data = Noise_data/(2**(bit_depth-1))


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
pass




