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
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')
gan = load_model('trained_wganlp_model_epoch_Dense_500')
#print(gan.summary())
N_fft = 1024

text_file = open("Noise_list.txt", "r")  #make sure this is at the same location as this file
lines = text_file.readlines()
text_file.close()
Noise_codebook2 = np.zeros((1024,9))

speech_file = open("Speech_list.txt", "r") #make sure this is at the same location as this file
speech_lines = speech_file.readlines()
speech_file.close()
Speech_codebook2 = np.zeros((1024,6))

for frequency_bin in range (0,len(lines)):
  string_list = lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   Noise_codebook2[frequency_bin,component] = float(string_list[component])

 

for frequency_bin in range (0,len(speech_lines)):
  string_list = speech_lines[frequency_bin].split()
  #print(string_list)
  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Speech_codebook2[frequency_bin,component] = 0
   else:
    Speech_codebook2[frequency_bin,component] = float(string_list[component])


directories = os.listdir('SNR_seg/-6dB')
directories2 = os.listdir('creating_dataset2/dataset_all4/noise/-6dB')
def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')
 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft





for No_of_data in range (0,10):
   samplerate, data = wavfile.read("SNR_seg/-6dB/"+ directories[No_of_data])
   Bit_Check = wave.open("SNR_seg/-6dB/"+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   audio= np.zeros(len(data))
   audio_GAN= np.zeros(len(data))
   for No_of_overlaps in range (0,Overlaps-5): #need to fix this, ignores the last few parts
     
     Rectangular_windowed_signal = data[0+128*No_of_overlaps:512+128*No_of_overlaps]
     Estimated_speech_PSD = np.zeros(N_fft)
     Estimated_noise_PSD = np.zeros(N_fft)
     GAN_estimate = np.zeros(N_fft)
     FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
     Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
     PSD_window_scaling = np.sum(Hann_window**2)
     PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)

     Tensor_PSD = tf.convert_to_tensor(PSD_of_windowed_signal.reshape(1,1024), tf.float32)
     
     Generated_codebook = gan(Tensor_PSD)
     #print(np.shape(Generated_codebook.numpy()))
     Generated_codebook = Generated_codebook.numpy()
     Generated_codebook_reshaped = np.abs((Generated_codebook.reshape(1024,9)))
     #print(np.shape(Generated_codebook_reshaped))
    #  myFile = open('GAN_list.txt', 'r+')
    #  np.savetxt(myFile, Generated_codebook_reshaped)
    #  myFile.close()

     Generated_codebook_inverse = np.linalg.pinv(Generated_codebook_reshaped, rcond=1e-15)
     Generated_coeffs = Generated_codebook_inverse*PSD_of_windowed_signal
     Generated_coeffs = np.transpose(Generated_coeffs)
     GAN_noise_codebook = (Generated_coeffs*Generated_codebook_reshaped)
     GAN_noise_codebook = GAN_noise_codebook.clip(min=0)
     #GAN_noise_codebook = GAN_noise_codebook.reshape(1024,9)
     #print(np.shape(GAN_noise_codebook))
     myFile = open('GAN_list.txt', 'r+')
     np.savetxt(myFile, GAN_noise_codebook)
     myFile.close()

     Noise_inverse = np.linalg.pinv(Noise_codebook2, rcond=1e-15)
     Noise_coeffs = Noise_inverse*PSD_of_windowed_signal
     Speech_inverse = np.linalg.pinv(Speech_codebook2, rcond=1e-15)
     Speech_coeffs = Speech_inverse*PSD_of_windowed_signal
     Speech_coeffs = np.transpose(Speech_coeffs)
     Noise_coeffs = np.transpose(Noise_coeffs)
     Estimated_speech_PSD_codebook = Speech_coeffs*Speech_codebook2
     Estimated_speech_PSD_codebook = Estimated_speech_PSD_codebook.clip(min=0)
     Estimated_noise_PSD_codebook = Noise_coeffs*Noise_codebook2
     Estimated_noise_PSD_codebook = Estimated_noise_PSD_codebook.clip(min=0)
     myFile = open('Noise_code.txt', 'r+')
     np.savetxt(myFile, Estimated_noise_PSD_codebook)
     myFile.close()
     
     


     for Freq_bin in range (0,N_fft):
       GAN_estimate[Freq_bin]=np.sum(GAN_noise_codebook[Freq_bin,:])
     #print(np.shape(GAN_estimate))
     
     for Freq_bin in range (0,N_fft):
       Estimated_speech_PSD[Freq_bin]=np.sum(Estimated_speech_PSD_codebook[Freq_bin,:])
       Estimated_noise_PSD[Freq_bin]=np.sum(Estimated_noise_PSD_codebook[Freq_bin,:])
     #print(np.shape(Estimated_noise_PSD))
     Noise_suppression = 2 #These three value control the balance between noise suppression and the quality of the recovered speech
     Speech_emphasis = 1  
     Weiner_scaling = 1


     Current_frame_weiner_coeffs = Estimated_speech_PSD/(Noise_suppression*Estimated_noise_PSD+Speech_emphasis*Estimated_speech_PSD)
     Current_frame_weiner_coeffs_GAN = Estimated_speech_PSD/(400*Noise_suppression*GAN_estimate+Speech_emphasis*Estimated_speech_PSD)
     De_noised_frame = (Current_frame_weiner_coeffs**Weiner_scaling)*FFT_of_windowed_signal
     De_noised_frame_GAN = (Current_frame_weiner_coeffs_GAN**Weiner_scaling)*FFT_of_windowed_signal
     FFT_to_audio = np.fft.ifft(De_noised_frame)
     FFT_to_audio_GAN = np.fft.ifft(De_noised_frame_GAN)
     audio[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio[0:512] #recover only the windowed signal and not the zero-pad
     audio_GAN[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio_GAN[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio_GAN[0:512] #recover only the windowed signal and not the zero-pad
   sf.write('MY_Experimenting_Folder/'+ directories[No_of_data], audio, 16000, 'PCM_16')
   sf.write('MY_Experimenting_Folder/GAN'+ directories[No_of_data], audio_GAN, 16000, 'PCM_16')    