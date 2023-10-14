import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
os.chdir('../..')

#ATTENTION!!!!! If switching from ploting the densities of noise to speech, please copy paste the folder which contains the previously generated
#plots to somewhere else. Basically, this file overwrites the existing plots. ATTENTION!!!

# Change to 'Others/Creating-Figures/LTA_PSD_SPEECH.txt' if the univariate GMM density of speech is desired.
speech_file = open("Others/Creating-Figures/LTA_PSD_NOISE.txt", "r") #make sure this is at the same location as this file
speech_lines = speech_file.readlines()
speech_file.close()


#Change to 'Models-Setup/GAN-Setup/Speech_GMM_Component_probabilities.txt' and similar if the GMMs of speech are required.
Means_file = open('Models-Setup/GAN-Setup/Noise_GMM_codebook.txt','r')
Means_lines = Means_file.readlines()
Means_file.close()

Covariances_file = open('Models-Setup/GAN-Setup/Noise_GMM_Component_variance.txt','r')
Covariances_lines = Covariances_file.readlines()
Covariances_file.close()

Component_probs_file = open('Models-Setup/GAN-Setup/Noise_GMM_Component_probabilities.txt','r')
Component_probs_lines = Component_probs_file.readlines()
Component_probs_file.close()

os.makedirs('Others/Creating-Figures/GMM_Density_Plots',exist_ok=True)

string_list = speech_lines[0].split()
Means_list = Means_lines[0].split()
Covariances_list = Covariances_lines[0].split()
Component_probs_list = Component_probs_lines[0].split()
Long_term_average_accross_all_data = np.zeros((len(speech_lines),len(string_list)))
Means = np.zeros((len(Means_lines),len(Means_list)))
Covariances = np.zeros((len(Covariances_lines),len(Covariances_list)))
Component_probs = np.zeros((len(Component_probs_lines),len(Component_probs_list)))

for frequency_bin in range (0,len(Means_lines)):
  Means_list = Means_lines[frequency_bin].split()
  for component in range (0, len(Means_list)):
   if Means_list[component] == "nan":
    Means[frequency_bin,component] = 0
   else:
    Means[frequency_bin,component] = float(Means_list[component])


for frequency_bin in range (0,len(Covariances_lines)):
  Covariances_list = Covariances_lines[frequency_bin].split()
  for component in range (0, len(Covariances_list)):
   if Covariances_list[component] == "nan":
    Covariances[frequency_bin,component] = 0
   else:
    Covariances[frequency_bin,component] = float(Covariances_list[component])


for frequency_bin in range (0,len(Component_probs_lines)):
  Component_probs_list = Component_probs_lines[frequency_bin].split()
  for component in range (0, len(Component_probs_list)):
   if Component_probs_list[component] == "nan":
    Component_probs[frequency_bin,component] = 0
   else:
    Component_probs[frequency_bin,component] = float(Component_probs_list[component])



for frequency_bin in range (0,len(speech_lines)):
  string_list = speech_lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Long_term_average_accross_all_data[frequency_bin,component] = 0
   else:
    Long_term_average_accross_all_data[frequency_bin,component] = float(string_list[component])



for frequency_bin in range (0,len(speech_lines)):
 Mean_PSD_per_frequency_bin = Long_term_average_accross_all_data[frequency_bin,:]
 GMM_Means_per_frequency_bin = Means[frequency_bin,:]
 GMM_Covariances_per_frequency_bin = Covariances[frequency_bin,:]
 GMM_Component_probs_per_frequency_bin = Component_probs[frequency_bin,:]
 Minimum_PSD_val = min(Mean_PSD_per_frequency_bin)
 Maximum_PSD_val = max(Mean_PSD_per_frequency_bin)

 x = np.linspace(Minimum_PSD_val-Minimum_PSD_val,Maximum_PSD_val, 10000)
 pdfs = [p * ss.norm.pdf(x, mu, sd) for mu, sd, p in zip(GMM_Means_per_frequency_bin, GMM_Covariances_per_frequency_bin, GMM_Component_probs_per_frequency_bin)]
 density = np.sum(np.array(pdfs), axis=0)
 #Adjust the figure size as required.
 plt.figure(figsize=(15,10))
 plt.plot(x,density)
 plt.plot(Mean_PSD_per_frequency_bin,np.zeros_like(Mean_PSD_per_frequency_bin),marker='+',c='blue',lw=0)
 plt.xlabel('LTA PSD')
 plt.ylabel("Density")
 plt.xscale('log')
 plt.yscale('log')
 plt.savefig('Others/Creating-Figures/GMM_Density_Plots/'+'Frequency_bin'+str(frequency_bin)+'_GMM_Density'+'.png')
 plt.close()
 pass


