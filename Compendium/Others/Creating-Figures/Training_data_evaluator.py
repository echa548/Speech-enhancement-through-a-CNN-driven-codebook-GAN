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
os.chdir('../..')

os.makedirs('Others/Creating-Figures/Training_data_evaluation',exist_ok=True)
os.makedirs('Others/Creating-Figures/Real_spectrum_stem_evaluation',exist_ok=True)
os.makedirs('Others/Creating-Figures/GAN_spectrum_stem_evaluation',exist_ok=True)
gan = tf.saved_model.load('Models/GAN-Models/Full_Curriculum_1000_generator')
real_noise_PSD = np.load('Models-Setup/GAN-Setup/Pure_Noise_PSDs.npy')
Mixture = np.load('Models-Setup/GAN-Setup/Noisy_Mixture_PSDs')
#This script checks the performance of the GAN using the training data.
row = np.shape(Mixture)[0]
num_samples = 1024
fstep = 16000/1024
f = np.linspace(0,(num_samples-1)*fstep, num_samples)
for noise_PSD in range (0,row):
 GAN_estimate = np.zeros(1024)
 Tensor_PSD = tf.convert_to_tensor(Mixture[noise_PSD,:].reshape(1,1024), tf.float32)
 Generated_codebook = gan(Tensor_PSD)
 Generated_codebook = Generated_codebook.numpy()
 Generated_codebook_reshaped = np.abs((Generated_codebook.reshape(1024,9)))
 #Marker comment
 Generated_codebook_inverse = np.linalg.pinv(Generated_codebook_reshaped, rcond=1e-15)
 Generated_coeffs = Generated_codebook_inverse*Mixture[noise_PSD,:]
 Generated_coeffs = np.transpose(Generated_coeffs)
 GAN_noise_codebook = (Generated_coeffs*Generated_codebook_reshaped)
 GAN_noise_codebook = GAN_noise_codebook.clip(min=0)
 #Marker comment
  #To apply method 2 please comment the designated marked section 'Marker comment' and uncomment the following:
  # Generated_codebook = Generated_codebook.numpy()
  # Generated_codebook_reshaped = np.abs((Generated_codebook.reshape(1024,9)))
  # shaped = Mixture[noise_PSD,:].reshape(1024,1)
  # sol = Noise_codebook2*shaped*Generated_codebook_reshaped
 #Uncomment only up to above.

#Marker comment
 for Freq_bin in range (0,1024):
  GAN_estimate[Freq_bin]=np.sum(GAN_noise_codebook[Freq_bin,:])
  pass
#Marker comment 

  #To apply method 2 please comment the designated marked section 'Marker comment' and uncomment the following: #Note that if using this method, please boost the resulting sum by a billion or more.
  # for Freq_bin in range (0,1024):
  #   GAN_estimate[Freq_bin]=np.sum(sol[Freq_bin,:])
   #Uncomment only up to above.

 plt.stem(f[0:512],2*GAN_estimate[0:512],'b',markerfmt = ' ',linefmt = 'black')
 plt.xlabel('Frequency(Hz)')
 plt.ylabel("Spectral power")
 plt.ylim(bottom=1e-20)
 plt.title('Stem plot of GAN spectrum')
 plt.savefig('Others/Creating-Figures/GAN_spectrum_stem_evaluation/'+'GAN_spectrum_Datapoint'+'_'+str(noise_PSD)+'.png')
 plt.close()

 plt.stem(f[0:512],2*GAN_estimate[0:512],'b',markerfmt = ' ',linefmt = 'black')
 plt.xlabel('Frequency(Hz)')
 plt.ylabel("Spectral power")
 plt.ylim(bottom=1e-20)
 plt.title('Stem plot of GAN spectrum')
 plt.savefig('Others/Creating-Figures/Real_spectrum_stem_evaluation/'+'Real_spectrum_Datapoint'+'_'+str(noise_PSD)+'.png')
 plt.close()

 plt.plot(f[0:512],2*GAN_estimate[0:512], color = 'green', label = 'WGAN-GP estimate')
 plt.plot(f[0:512],2*real_noise_PSD[noise_PSD,:][0:512],)
 plt.xlabel('Frequency(Hz)')
 plt.ylabel("Power Spectral Density (dB/Hz)")
 plt.xscale('log')
 plt.yscale('log')
 plt.ylim(bottom=1e-20)
 plt.legend()
 plt.xlabel('Frequency (Hz)')
 plt.ylabel("Power Spectral Density (dB/Hz)")
 plt.title('Candidate spectrum')
 plt.legend()
 plt.savefig('Others/Creating-Figures/Training_data_evaluation/'+'Datapoint'+'_'+str(noise_PSD)+'.png')
 plt.close()
 pass
