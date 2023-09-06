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
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
Noise_gan = tf.saved_model.load('Noise_PSD_Generator_epoch_73')
Speech_gan = tf.saved_model.load('trained_wganlp_model_epoch_Dense_4_generator')
N_fft = 1024

text_file = open("Noise_list.txt", "r")  #make sure this is at the same location as this file
lines = text_file.readlines()
text_file.close()
Noise_codebook2 = np.zeros((1024,9))

speech_file = open("Modified_speech_list.txt", "r") #make sure this is at the same location as this file
speech_lines = speech_file.readlines()
speech_file.close()
Speech_codebook2 = np.zeros((1024,6))


for frequency_bin in range (0,len(lines)):
  string_list = lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   Noise_codebook2[frequency_bin,component] = float(string_list[component])

 

for frequency_bin in range (0,len(speech_lines)):
  string_list = speech_lines[frequency_bin].split()

  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Speech_codebook2[frequency_bin,component] = 0
   else:
    Speech_codebook2[frequency_bin,component] = float(string_list[component])


directories = os.listdir('SNR_seg/-6dB')

def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')

 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft


#Do for all SNRs making graphs and folders for each filtered file and also output the GMM file.
No_of_data_to_filter = 20
num_samples = 1024
fstep = 16000/1024
f = np.linspace(0,(num_samples-1)*fstep, num_samples)





#targets = [9,6,3,0,-3,-6,-9]
targets = [-9]
for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/Noise_PSDs',directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/Speech_PSDs',directory) 
 os.makedirs(path, exist_ok=True)
 
for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/Wiener',directory) 
 os.makedirs(path, exist_ok=True)


for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/GAN',directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/GAN_Noise_only',directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/Perfect',directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join('Filter_outputs/GAN_Noise_GMM_speech',directory) 
 os.makedirs(path, exist_ok=True)


for target_db in range (0,len(targets)):

 Path_of_noisy_mixture = 'SNR_seg/'+str(targets[target_db])+'dB'+'/'
 Path_of_real_noise_files = 'SNR_seg/noise/'+str(targets[target_db])+'dB'+'/'
 Noise_files = os.listdir('SNR_seg/noise/'+str(targets[target_db])+'dB')
 Noisy_mixture_files = os.listdir('SNR_seg/'+str(targets[target_db])+'dB')
 Path_of_real_speech_files = 'SNR_seg/clean/'



 for No_of_data in range (0,No_of_data_to_filter):
   samplerate, data = wavfile.read(Path_of_noisy_mixture+ directories[No_of_data])
   Bit_Check = wave.open(Path_of_noisy_mixture+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   audio= np.zeros(len(data))
   audio_GAN= np.zeros(len(data))
   audio_GAN_Noise_only= np.zeros(len(data))
   audio_perfect= np.zeros(len(data))
   samplerate, real_noise = wavfile.read(Path_of_real_noise_files+ directories[No_of_data])
   Bit_Check = wave.open(Path_of_real_noise_files+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   real_noise = real_noise/(2**(bit_depth-1))

   samplerate, real_speech = wavfile.read(Path_of_real_speech_files+ directories[No_of_data])
   Bit_Check = wave.open(Path_of_real_speech_files+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   real_speech = real_speech/(2**(bit_depth-1))

   for No_of_overlaps in range (0,Overlaps-5): #need to fix this, ignores the last few parts
     
     Rectangular_windowed_signal = data[0+128*No_of_overlaps:512+128*No_of_overlaps]
     Estimated_speech_PSD = np.zeros(N_fft)
     Estimated_noise_PSD = np.zeros(N_fft)
     GAN_noise_estimate = np.zeros(N_fft)
     GAN_speech_estimate = np.zeros(N_fft)
     FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)

     Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
     PSD_window_scaling = np.sum(Hann_window**2)
     PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)



     Tensor_PSD = tf.convert_to_tensor(PSD_of_windowed_signal.reshape(1,1024), tf.float32)
     
     Generated_codebook = Noise_gan(Tensor_PSD)
     Generated_codebook = Generated_codebook.numpy()
     Generated_codebook_reshaped = np.abs((Generated_codebook.reshape(1024,9)))
     
     Generated_speech_codebook = Speech_gan(Tensor_PSD)
     Generated_speech_codebook = Generated_speech_codebook.numpy()
     Generated_speech_codebook_reshaped = np.abs((Generated_speech_codebook.reshape(1024,6))) #TODO

     Generated_codebook_inverse = np.linalg.pinv(Generated_codebook_reshaped, rcond=1e-15)
     Generated_coeffs = Generated_codebook_inverse*PSD_of_windowed_signal
     Generated_coeffs = np.transpose(Generated_coeffs)
     GAN_noise_codebook = (Generated_coeffs*Generated_codebook_reshaped)
     GAN_noise_codebook = GAN_noise_codebook.clip(min=0)

     Generated_speech_codebook_inverse = np.linalg.pinv(Generated_speech_codebook_reshaped, rcond=1e-15)
     Generated_speech_coeffs = Generated_speech_codebook_inverse*PSD_of_windowed_signal
     Generated_speech_coeffs = np.transpose(Generated_speech_coeffs)
     GAN_speech_codebook = (Generated_speech_coeffs*Generated_speech_codebook_reshaped)
     GAN_speech_codebook = GAN_speech_codebook.clip(min=0)


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

     
     Rectangular_windowed_real_signal = real_noise[0+128*No_of_overlaps:512+128*No_of_overlaps]
     FFT_of_real_windowed_signal = Hann_window_a_signal(Rectangular_windowed_real_signal)
     PSD_of_real_windowed_signal = (np.abs(FFT_of_real_windowed_signal)**2)/(samplerate*PSD_window_scaling)

     Rectangular_windowed_real_speech_signal = real_speech[0+128*No_of_overlaps:512+128*No_of_overlaps]
     FFT_of_real_windowed_speech_signal = Hann_window_a_signal(Rectangular_windowed_real_speech_signal)
     PSD_of_real_windowed_speech_signal = (np.abs(FFT_of_real_windowed_speech_signal)**2)/(samplerate*PSD_window_scaling)


     for Freq_bin in range (0,N_fft):
       GAN_noise_estimate[Freq_bin]=np.sum(GAN_noise_codebook[Freq_bin,:])
 
     for Freq_bin in range (0,N_fft):
       GAN_speech_estimate[Freq_bin]=np.sum(GAN_speech_codebook[Freq_bin,:])

     for Freq_bin in range (0,N_fft):
       Estimated_speech_PSD[Freq_bin]=np.sum(Estimated_speech_PSD_codebook[Freq_bin,:])
       Estimated_noise_PSD[Freq_bin]=np.sum(Estimated_noise_PSD_codebook[Freq_bin,:])
     
     Noise_suppression = 1 #These three value control the balance between noise suppression and the quality of the recovered speech
     Speech_emphasis = 1  
     Weiner_scaling = 1

  

     GAN_noise_estimate[512:1024]=np.flip(GAN_noise_estimate[0:512])
     Estimated_speech_PSD[512:1024]= np.flip(Estimated_speech_PSD[0:512])
     GAN_speech_estimate[512:1024] = np.flip(GAN_speech_estimate[0:512])
     scalar_factor_noise = sum([psd_value for psd_value in GAN_noise_estimate]) 
     scalar_factor_speech = sum([psd_value for psd_value in GAN_speech_estimate])
     GAN_noise_estimate_normalised = GAN_noise_estimate/scalar_factor_noise
     GAN_speech_estimate_normalised = GAN_speech_estimate/scalar_factor_speech
     
     
     Estimated_noise_PSD[512:1024]= np.flip(Estimated_noise_PSD[0:512])
     scalar_factor_noise = sum([psd_value for psd_value in Estimated_noise_PSD])
     Estimated_noise_PSD_normalised = Estimated_noise_PSD/scalar_factor_noise

     Estimated_speech_PSD[512:1024]= np.flip(Estimated_speech_PSD[0:512])
     scalar_factor_speech = sum([psd_value for psd_value in Estimated_speech_PSD])
     Estimated_speech_PSD_normalised = Estimated_speech_PSD/scalar_factor_speech

     PSD_of_real_windowed_signal[512:1024]= np.flip(PSD_of_real_windowed_signal[0:512])
     scalar_factor_noise = sum([psd_value for psd_value in PSD_of_real_windowed_signal])
     normalised_noise = PSD_of_real_windowed_signal/scalar_factor_noise
     
     PSD_of_real_windowed_speech_signal[512:1024] = np.flip(PSD_of_real_windowed_speech_signal[0:512])
     scalar_factor_speech = sum([psd_value for psd_value in PSD_of_real_windowed_speech_signal])
     PSD_of_real_windowed_speech_signal_normalised = PSD_of_real_windowed_speech_signal/scalar_factor_speech

     PSD_of_windowed_signal[512:1024]= np.flip(PSD_of_windowed_signal[0:512])
     scalar_wat = sum([psd_value for psd_value in PSD_of_windowed_signal])/2
     PSD_of_estimate = PSD_of_windowed_signal/scalar_wat
     

     Current_frame_weiner_coeffs = Estimated_speech_PSD_normalised/(Noise_suppression*Estimated_noise_PSD_normalised+Speech_emphasis*Estimated_speech_PSD_normalised)
     Current_frame_weiner_coeffs_GAN =GAN_speech_estimate_normalised/(GAN_noise_estimate_normalised+GAN_speech_estimate_normalised)
     Current_frame_weiner_coeffs_GAN_noise_GMM_speech = Estimated_speech_PSD_normalised/(GAN_noise_estimate_normalised+Estimated_speech_PSD_normalised)
     Current_frame_weiner_coeffs_perfect = PSD_of_real_windowed_speech_signal_normalised/(normalised_noise+PSD_of_real_windowed_speech_signal_normalised)

     De_noised_frame = (Current_frame_weiner_coeffs**Weiner_scaling)*FFT_of_windowed_signal
     De_noised_frame_GAN = (Current_frame_weiner_coeffs_GAN**Weiner_scaling)*FFT_of_windowed_signal
     De_noised_frame_perfect = Current_frame_weiner_coeffs_perfect*FFT_of_windowed_signal
     De_noised_frame_GAN_noise_GMM_speech = Current_frame_weiner_coeffs_GAN_noise_GMM_speech*FFT_of_windowed_signal

     FFT_to_audio = np.fft.ifft(De_noised_frame)
     FFT_to_audio_GAN = np.fft.ifft(De_noised_frame_GAN)
     FFT_to_audio_GAN_noise_GMM_speech = np.fft.ifft(De_noised_frame_GAN_noise_GMM_speech)
     FFT_to_audio_perfect = np.fft.ifft(De_noised_frame_perfect)


     audio[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio[0:512] #recover only the windowed signal and not the zero-pad
     audio_GAN[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio_GAN[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio_GAN[0:512] #recover only the windowed signal and not the zero-pad
     audio_GAN_Noise_only[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio_GAN_Noise_only[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio_GAN_noise_GMM_speech[0:512]
     audio_perfect[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio_perfect[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio_perfect[0:512] #recover only the windowed signal and not the zero-pad
     

    
    #  plt.plot(f[0:512],155*GAN_noise_estimate[0:512], label = 'GAN noise')
    #  plt.plot(f[0:512],2*PSD_of_real_windowed_signal[0:512], label = 'Real noise')
    #  plt.plot(f[0:512],2*PSD_of_windowed_signal[0:512], label = 'Mixture')
    #  plt.xlabel('log Frequency (Hz)')
    #  plt.ylabel("log Power Spectral Density (dB/Hz)")
    #  plt.title('GAN vs Real vs GMM noise PSDs')
    #  plt.xscale('log')
    #  plt.yscale('log')
    #  plt.legend()
    #  plt.ylim(bottom=1e-20)
    #  plt.savefig('Filter_outputs/Noise_PSDs/'+str(targets[target_db])+'dB/'+directories[No_of_data]+'_'+'Frame'+str(No_of_overlaps)+'.png')
    #  plt.close()

    
     
     #plt.plot(f[0:512],2*ESt[0:512], label = 'GAN speech')
    #  plt.plot(f[0:512],2*PSD_of_real_windowed_speech_signal[0:512], label = 'Real speech')
    #  plt.plot(f[0:512],2*Estimated_speech_PSD[0:512], label = 'GMM speech')
    #  plt.plot(f[0:512],2*PSD_of_windowed_signal[0:512], label = 'Mixture')
    #  plt.xlabel('log Frequency (Hz)')
    #  plt.ylabel("log Power Spectral Density (dB/Hz)")
    #  plt.title('GAN vs Real vs GMM speech PSDs')
    #  plt.xscale('log')
    #  plt.yscale('log')
    #  plt.legend()
    #  plt.ylim(bottom=1e-20)
    #  plt.savefig('Filter_outputs/Speech_PSDs/'+str(targets[target_db])+'dB/'+directories[No_of_data]+'_'+'Frame'+str(No_of_overlaps)+'.png')
    #  plt.close()

    #  PSD_of_windowed_signal= PSD_of_windowed_signal[0:512]
    
    #  plt.plot(f[0:512],2*PSD_of_windowed_signal[0:512], label = 'Mixture')
    #  #plt.plot(f[0:512],2*PSD_of_real_windowed_signal[0:512]+2*PSD_of_real_windowed_speech_signal[0:512], label = 'Uncorrelated')
    #  #plt.plot(f[0:512],2*PSD_of_real_windowed_signal[0:512], label = 'Real noise')
    #  plt.plot(f[0:512],2*PSD_of_real_windowed_speech_signal[0:512], label = 'Real speech')
    # #  plt.plot(f[0:512],200*GAN_noise_estimate[0:512], label = 'GAN noise')
    #  plt.plot(f[0:512], (2*PSD_of_windowed_signal[0:512]-200*GAN_noise_estimate[0:512]).clip(min=0)/100,label = 'Estimated speech')
    #  plt.xlabel('log Frequency (Hz)')
    #  plt.ylabel("log Power Spectral Density (dB/Hz)")
    #  plt.title('GAN vs Real vs GMM speech PSDs')
    #  plt.xscale('log')
    #  plt.yscale('log')
    #  plt.legend()
    #  plt.ylim(bottom=1e-20)
    #  plt.savefig('Filter_outputs/Speech_PSDs/'+str(targets[target_db])+'dB/'+directories[No_of_data]+'_'+'Frame'+str(No_of_overlaps)+'.png')
    #  plt.close()

   sf.write('Filter_outputs/Wiener/'+str(targets[target_db])+'dB/'+directories[No_of_data], audio, 16000, 'PCM_16')
   sf.write('Filter_outputs/GAN/'+str(targets[target_db])+'dB/'+directories[No_of_data], audio_GAN, 16000, 'PCM_16')
   sf.write('Filter_outputs/GAN_Noise_GMM_speech/'+str(targets[target_db])+'dB/'+directories[No_of_data], audio_GAN_Noise_only, 16000, 'PCM_16')
   sf.write('Filter_outputs/Perfect/'+str(targets[target_db])+'dB/'+directories[No_of_data], audio_perfect, 16000, 'PCM_16')


pass
