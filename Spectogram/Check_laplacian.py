import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
from scipy.stats import expon
from scipy.optimize import curve_fit
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')

speech_file = open("Mean_check.txt", "r") #make sure this is at the same location as this file
speech_lines = speech_file.readlines()
speech_file.close()
string_list = speech_lines[0].split()


Means_file = open('Speech_list.txt','r')
Means_lines = Means_file.readlines()
Means_file.close()

index_logger = []
Means_list = Means_lines[0].split()
Means = np.zeros((len(Means_lines),len(Means_list)))

Long_term_average_accross_all_data = np.zeros((len(speech_lines),len(string_list)))
for frequency_bin in range (0,len(speech_lines)):
  string_list = Means_lines[frequency_bin].split()
  trigger = 1
  for component in range (0, len(string_list)):
   if string_list[component] == "nan" and trigger ==1:
    index_logger.append(frequency_bin)
    trigger =0

for frequency_bin in range (0,len(speech_lines)):
  string_list = speech_lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Long_term_average_accross_all_data[frequency_bin,component] = 0
   else:
    Long_term_average_accross_all_data[frequency_bin,component] = float(string_list[component])

    
for frequency_bin in range (0,len(Means_lines)):
  Means_list = Means_lines[frequency_bin].split()
  for component in range (0, len(Means_list)):
   if Means_list[component] == "nan":
    Means[frequency_bin,component] = 0
   else:
    Means[frequency_bin,component] = float(Means_list[component])
#fit the laplacian
index_logger = np.asarray(index_logger)

def neg_exponential(x, scale):
    return expon.pdf(x, scale=scale)


Modified_means = Means

for nan_row in range (0, len(index_logger)):
 data = Long_term_average_accross_all_data[index_logger[nan_row],:]
 hist_values, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.6, color='b')
 scale_initial_guess = 1.0
 params, covariance = curve_fit(neg_exponential, xdata=bins[:-1], ydata=hist_values, p0=scale_initial_guess)
 fitted_scale = params[0]
 fitted_mean = np.mean(data)
 x = np.linspace(min(data), max(data), 4500)
 plt.plot(x, neg_exponential(x, fitted_scale), 'r')
 plt.savefig('Non_converging/'+'Frequency_bin_'+'Frequency_bin'+str(index_logger[nan_row])+'.png')
 plt.close()
 Modified_means[index_logger[nan_row],:] = fitted_mean
 
myFile = open('Modified_speech_list.txt', 'r+')
np.savetxt(myFile, Modified_means)
myFile.close()
