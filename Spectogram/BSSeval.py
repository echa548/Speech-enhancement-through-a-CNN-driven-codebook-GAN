import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import fast_bss_eval
from scipy.io import wavfile
import tensorflow as tf
import torchaudio
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')

directories_of_GAN_estimate  = os.listdir('MY_Experimenting_Folder/GAN_estimate')
directories_of_Wiener_filtered  = os.listdir('MY_Experimenting_Folder/Wiener_filtered')
Path_to_Wiener_filtered_data ='MY_Experimenting_Folder/Wiener_filtered/'
Path_to_GAN_estimate_data ='MY_Experimenting_Folder/GAN_estimate/'
Path_to_Reference = 'creating_dataset2/dataset_all4/clean/'
Score_list=[]
for filename in range (0,len(directories_of_Wiener_filtered)):
 ref, fs = torchaudio.load(Path_to_Reference+directories_of_Wiener_filtered[filename])
 est, _ = torchaudio.load(Path_to_Wiener_filtered_data+directories_of_Wiener_filtered[filename])
 est2, _ = torchaudio.load(Path_to_GAN_estimate_data+directories_of_GAN_estimate[filename])
 Wsdr, Wsir, Wsar = fast_bss_eval.si_bss_eval_sources(ref, est, zero_mean=False, clamp_db=None, compute_permutation=False, load_diag=None)
 Gsdr, Gsir, Gsar = fast_bss_eval.si_bss_eval_sources(ref, est2, zero_mean=False, clamp_db=None, compute_permutation=False, load_diag=None)
 Score_list.append('SDR:'+str(Wsdr.numpy())+'dB'+' '+'SIR:'+str(Wsir.numpy())+'dB'+' '+'SAR:'+str(Wsar.numpy())+'dB'+' '+'File:'+str(directories_of_Wiener_filtered[filename]))
 Score_list.append('SDR:'+str(Gsdr.numpy())+'dB'+' '+'SIR:'+str(Gsir.numpy())+'dB'+' '+'SAR:'+str(Gsar.numpy())+'dB'+' '+'File:'+str(directories_of_GAN_estimate[filename]))
 Score_list.append('SDR Improvement:'+str(Gsdr.numpy()-Wsdr.numpy())+'dB'+' '+'SIR Improvement:'+str(Gsir.numpy()-Wsir.numpy())+'dB'+' '+'SAR Improvement:'+str(Gsar.numpy()-Wsar.numpy())+'dB')

 pass


with open ("Overall_scores.txt",'w') as f:
       for scores in Score_list:
        f.write(f"{scores}\n")

         