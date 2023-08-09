import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')

speech_file = open("Noise_check.txt", "r") #make sure this is at the same location as this file
speech_lines = speech_file.readlines()
speech_file.close()



Means_file = open('Noise_list.txt','r')
Means_lines = Means_file.readlines()
Means_file.close()

Covariances_file = open('Noise_sigma.txt','r')
Covariances_lines = Covariances_file.readlines()
Covariances_file.close()

Component_probs_file = open('Noise_probs.txt','r')
Component_probs_lines = Component_probs_file.readlines()
Component_probs_file.close()


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

#print(np.shape(Long_term_average_accross_all_data))


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



# for frequency_bin in range (0,len(speech_lines)):
#  Mean_PSD_per_frequency_bin = Long_term_average_accross_all_data[frequency_bin,:]
#  GMM_Means_per_frequency_bin = Means[frequency_bin,:]
#  GMM_Covariances_per_frequency_bin = Covariances[frequency_bin,:]
#  #GMM_Component_probs_per_frequency_bin = Component_probs[frequency_bin,:]
#  samples = []
#  #Index_choices = int(np.linspace(0,len(GMM_Means_per_frequency_bin)-1,len(GMM_Means_per_frequency_bin)))
#  #print(Index_choices)
#  How_many_u_want_HUH = 100000
#  for i in range(How_many_u_want_HUH):
#   Random_index = np.random.choice([0,1,2,3,4,5])
#   samples.append(np.random.normal(GMM_Means_per_frequency_bin[Random_index], GMM_Covariances_per_frequency_bin[Random_index], 1))
#  #sns.distplot(samples, hist=False)
#  #plt.show()
#  samples = np.asarray(samples)
#  df = pd.DataFrame(samples).melt(value_name = 'PSD_val')
#  sns.displot(data=df, x ='PSD_val', kind="kde")
#  plt.xlim(0,max(Mean_PSD_per_frequency_bin))
#  #plt.show()
#  plt.savefig('sea/'+'Frequency_bin_'+'Frequency_bin'+str(frequency_bin)+'.png',dpi=300)
#  plt.close()

#  pass



for frequency_bin in range (0,len(speech_lines)):
 Mean_PSD_per_frequency_bin = Long_term_average_accross_all_data[frequency_bin,:]
 GMM_Means_per_frequency_bin = Means[frequency_bin,:]
 GMM_Covariances_per_frequency_bin = Covariances[frequency_bin,:]
 GMM_Component_probs_per_frequency_bin = Component_probs[frequency_bin,:]
 #print(np.sum(GMM_Component_probs_per_frequency_bin))
 #x = np.arange(-5., 5., 0.01)
 Minimum_PSD_val = min(Mean_PSD_per_frequency_bin)
 Maximum_PSD_val = max(Mean_PSD_per_frequency_bin)
 #print(Minimum_PSD_val)
 #print(Maximum_PSD_val)
 x = np.linspace(Minimum_PSD_val-Minimum_PSD_val,Maximum_PSD_val, 10000)
 pdfs = [p * ss.norm.pdf(x, mu, sd) for mu, sd, p in zip(GMM_Means_per_frequency_bin, GMM_Covariances_per_frequency_bin, GMM_Component_probs_per_frequency_bin)]
 density = np.sum(np.array(pdfs), axis=0)
 #density = density/max(density) #small variance causes it to blow up, this prevents that.
 plt.figure(figsize=(15,10))
 plt.plot(x,density)
 plt.plot(Mean_PSD_per_frequency_bin,np.zeros_like(Mean_PSD_per_frequency_bin),marker='+',c='blue',lw=0)
 plt.xlabel('LTA PSD')
 plt.ylabel("Component probability")
 plt.savefig('GMM_noise_fit/'+'Frequency_bin_'+'Frequency_bin'+str(frequency_bin)+'.png')
 plt.close()
 pass


 
    #     plt.plot(log_likelihood_history)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Log Likelihood")
    #     plt.savefig('Speech_FFT_Log_Likelihood/'+'Frequency_bin_'+str(Frequency_bin)+'.png')
    #     plt.close()