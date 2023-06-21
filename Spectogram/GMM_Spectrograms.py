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
import os
from pydub import AudioSegment, effects
import wave
os.chdir('C:/Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram')

samplerate =0

def em(data, components, iterations, RNGseed):
 Number_of_samples = data.shape[0]
 np.random.seed(RNGseed)
 mean = np.random.rand(components)
 variance = np.random.rand(components)
 components_probs = np.random.dirichlet(np.ones(components))
 
 for em in range(iterations): 
  #responsibilities = np.zeros((Number_of_samples,components))
  #for i in range(Number_of_samples):
   #for component in range (components):
    #responsibilities[i,component] = components_probs[c] *\
    #tfp.distributions.distributions.Normal(loc= mean[component],scale = variance[component].prob(data[i]))
  responsibilities=tfp.distributions.Normal(loc=mean, scale = variance).prob(data.reshape(-1,1)).numpy()*components_probs #Adversarial network will attack here
  responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True) #Adversarial network will attack here
  components_responsibilities = np.sum(responsibilities, axis=0) #KEEP THIS!
  #Adversarial network will not need iterations like this code does.
 
  for component in range(components):
   components_probs[component] = components_responsibilities[component]/Number_of_samples
   mean[component] = np.sum(responsibilities[:,component]* data, )/components_responsibilities[component]
   variance[component] = np.sqrt(np.sum(responsibilities[:,component]*(data-mean[component])**2))/components_responsibilities[component]
 return mean, variance, components_probs

def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_window_fft = np.fft.fft(Hann_window,1024)
 #print(Hann_window_fft)
 Windowed_data_fft = np.fft.fft(Windowed_data,1024)
 Multiplication_in_Freq = Hann_window_fft*Windowed_data_fft
 #print(np.shape(Multiplication_in_Freq))
 return Multiplication_in_Freq
def trainGMMspeech():
 
 pass

def trainGMMnoise(Components,iterations,seed):
 directories = os.listdir('MY_Experimenting_Folder/Processed_audio/processed_noise')
 N_fft = 1024
 pre_codebook_array = np.zeros((N_fft,len(directories)))
 for No_of_data in range (0,len(directories)):
   samplerate, data = wavfile.read("MY_Experimenting_Folder/Processed_audio/processed_noise/"+ directories[No_of_data])
   Rectangular_windowed_signal = data[0:1023]
   FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
   Frequency_spacing = 1/(1024*samplerate)
   PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/2*Frequency_spacing
   
   for Frequency_bin in range (0,np.size(PSD_of_windowed_signal)):
    pre_codebook_array[Frequency_bin,No_of_data] = PSD_of_windowed_signal[Frequency_bin]
 for Frequency_bin in range (0,N_fft):
    Frequency_bin_accross_all_data = pre_codebook_array[Frequency_bin,:] 
    mean_vector, variance, component_probabilities = em(Frequency_bin_accross_all_data,Components,iterations,seed)
 pass


#True clusters
N_clusters = 18
iterations = 100
#RNG seed
seed = 21
trainGMMnoise(N_clusters,iterations,seed)


#Tensor representation of each image with only Red channel




