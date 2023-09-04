import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
import silence_tensorflow.auto

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
gan = tf.saved_model.load('trained_wganlp_model_epoch_Dense_14_generator')

real_noise_PSD = np.load('Clean_PSD_speech.npy')
Mixture = np.load('Mixture_PSD_speech.npy')

row = np.shape(Mixture)[0]
num_samples = 1024
fstep = 16000/1024
f = np.linspace(0,(num_samples-1)*fstep, num_samples)
for noise_PSD in range (0,row):
 GAN_estimate = np.zeros(1024)
 Tensor_PSD = tf.convert_to_tensor(Mixture[noise_PSD,:].reshape(1,1024), tf.float32)
 Generated_codebook = gan(Tensor_PSD)
 #print(np.shape(Generated_codebook.numpy()))
 Generated_codebook = Generated_codebook.numpy()
 Generated_codebook_reshaped = np.abs((Generated_codebook.reshape(1024,15)))
 Generated_codebook_inverse = np.linalg.pinv(Generated_codebook_reshaped, rcond=1e-15)
 Generated_coeffs = Generated_codebook_inverse*Mixture[noise_PSD,:]
 Generated_coeffs = np.transpose(Generated_coeffs)
 GAN_noise_codebook = (Generated_coeffs*Generated_codebook_reshaped)
 GAN_noise_codebook = GAN_noise_codebook.clip(min=0)
 for Freq_bin in range (0,1024):
  GAN_estimate[Freq_bin]=np.sum(GAN_noise_codebook[Freq_bin,:])
  pass
 
#  #plt.stem(f,10*np.log10(GAN_estimate),'b',markerfmt = ' ',linefmt = 'black')
#  plt.plot(f[0:512],2*GAN_estimate[0:512])
#  #plt.stem(f,real_noise_PSD[noise_PSD,:],'g',markerfmt = ' ', label = 'Real', linefmt = '--')
#  plt.xlabel('Frequency(Hz)')
#  plt.ylabel("Power co-efficient")
#  plt.xscale('log')
#  plt.yscale('log')
#  plt.ylim(bottom=1e-20)
#  #plt.legend()
#  plt.xlabel('Frequency (Hz)')
#  plt.ylabel("Power Spectral Density (dB/Hz)")
#  plt.title('Predicted GAN PSD')
#  plt.savefig('PSD_check_GAN/'+'GAN'+'_'+str(noise_PSD)+'.png')
#  plt.close()
 
#  #plt.stem(f,10*np.log10(real_noise_PSD[noise_PSD,:]),'b',markerfmt = ' ',linefmt = 'black')
#  plt.plot(f[0:512],2*real_noise_PSD[noise_PSD,:][0:512])
#  plt.xlabel('Frequency(Hz)')
#  plt.ylabel("Power Spectral Density (dB/Hz)")
#  plt.xscale('log')
#  plt.yscale('log')
#  plt.ylim(bottom=1e-20)
#  #plt.legend()
#  plt.xlabel('Frequency (Hz)')
#  plt.ylabel("Power Spectral Density (dB/Hz)")
#  plt.title('Real PSD')
#  plt.savefig('PSD_check_real/'+'Real'+'_'+str(noise_PSD)+'.png')
#  plt.close()

 plt.plot(f[0:512],2*GAN_estimate[0:512], color = 'green', label = 'GAN')
 plt.plot(f[0:512],2*real_noise_PSD[noise_PSD,:][0:512], label = 'Real')
 plt.xlabel('Frequency(Hz)')
 plt.ylabel("Power Spectral Density (dB/Hz)")
 plt.xscale('log')
 plt.yscale('log')
 plt.ylim(bottom=1e-20)
 #plt.legend()
 plt.xlabel('Frequency (Hz)')
 plt.ylabel("Power Spectral Density (dB/Hz)")
 plt.title('Real PSD')
 plt.legend()
 plt.savefig('PSD_combined/'+'Combined'+'_'+str(noise_PSD)+'.png')
 plt.close()
 pass
