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

from scipy.special import logsumexp
os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')
# print(tf.__version__)
# print(len(tf.config.list_physical_devices('GPU')))
samplerate =0
train_noise = 0
train_speech = 0
Path_to_noise = 'MY_Experimenting_Folder/Processed_audio/processed_noise'
Noise_file_for_filter = 'Noise_list.txt'
Path_to_noisy_speech = 'creating_dataset2/dataset_all4/-6dB'
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
  #Adversarial network will attack here
  responsibilities=tfp.distributions.Normal(loc=mean, scale = variance).prob(data.reshape(-1,1)).numpy()*components_probs 
  #Adversarial network will attack here
  #print(responsibilities)
  responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True) 
  components_responsibilities = np.sum(responsibilities, axis=0) #KEEP THIS!
  #Adversarial network will not need iterations like this code does.
  
  for component in range(components):
   components_probs[component] = components_responsibilities[component]/Number_of_samples
   mean[component] = np.sum(responsibilities[:,component]* data, )/components_responsibilities[component]
   variance[component] = np.sqrt(np.sum(responsibilities[:,component]*(data-mean[component])**2))/components_responsibilities[component]
 return mean, variance, components_probs

def em_with_guesses(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    log_likelihood_history = []

    for em_iter in range(n_iterations):
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
            )

        # Calculate the marginal log likelihood
        log_likelihood = np.sum(
            logsumexp(
                np.log(class_probs)
                +
                tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                    dataset.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        log_likelihood_history.append(log_likelihood)
    
    return class_probs, mus, sigmas, log_likelihood_history

def em_sieved(
    dataset,
    n_classes,
    n_iterations_pre_sieving,
    n_candidates,
    n_iterations_post_sieving,
    n_chosen_ones,
    random_seed,
):

    # (1) Pre-Sieving

    mus_list = []
    sigmas_list = []
    class_probs_list = []
    log_likelihood_history_list = []

    for candidate_id in range(n_candidates):
        np.random.seed(random_seed + candidate_id)

        mus = np.random.rand(n_classes)
        sigmas = np.random.rand(n_classes)
        class_probs = np.random.dirichlet(np.ones(n_classes))

        class_probs, mus, sigmas, log_likelihood_history = em_with_guesses(
            dataset,
            n_iterations_pre_sieving,
            class_probs,
            mus,
            sigmas,
        )
        mus_list.append(mus)
        sigmas_list.append(sigmas)
        class_probs_list.append(class_probs)
        log_likelihood_history_list.append(log_likelihood_history)
    
    # (2) Sieving, select the best candidates
    log_likelihood_history_array = np.array(log_likelihood_history_list)

    # Sort in descending order
    ordered_candidate_ids = np.argsort( - log_likelihood_history_array[:, -1])
    chosen_ones_ids = ordered_candidate_ids[:n_chosen_ones]

    # (3) Post-Sieving
    mus_chosen_ones_list = []
    sigmas_chosen_ones_list = []
    class_probs_chosen_ones_list = []
    log_likelihood_history_chosen_ones_list = []
    for chosen_one_id in chosen_ones_ids:
        class_probs, mus, sigmas, log_likelihood_history = em_with_guesses(
            dataset,
            n_iterations_post_sieving,
            class_probs_list[chosen_one_id],
            mus_list[chosen_one_id],
            sigmas_list[chosen_one_id],
        )

        mus_chosen_ones_list.append(mus)
        sigmas_chosen_ones_list.append(sigmas)
        class_probs_chosen_ones_list.append(class_probs)
        log_likelihood_history_chosen_ones_list.append(log_likelihood_history)
    
    # (4) Select the very best candidate
    log_likelihood_history_chosen_ones_array = np.array(log_likelihood_history_chosen_ones_list)

    # Sort in descending order
    ordered_chosen_ones_ids = np.argsort( - log_likelihood_history_chosen_ones_array[:, -1])

    best_chosen_one_id = ordered_chosen_ones_ids[0]
    best_mus = mus_chosen_ones_list[best_chosen_one_id]
    best_sigmas = sigmas_chosen_ones_list[best_chosen_one_id]
    best_class_probs = class_probs_chosen_ones_list[best_chosen_one_id]

    return best_class_probs, best_mus, best_sigmas, log_likelihood_history_chosen_ones_array

def Hann_window_a_signal(Windowed_data):
 Hann_window = sps.windows.hann(len(Windowed_data))
 Hann_Windowed_data = Hann_window*Windowed_data
 padded_signal = np.pad(Hann_Windowed_data,(0,512), 'constant')
 Windowed_data_fft = np.fft.fft(padded_signal,1024)
 return Windowed_data_fft


def trainGMMspeech(Components,iterations,seed):
 directories = os.listdir('Proper_speech')
 path_to_VAD_speech = 'Proper_speech/'
 N_fft = 1024
 pre_codebook_array = np.zeros((N_fft,len(directories)))
 speech_codebook_array = np.zeros((N_fft,Components))
 speech_CompProb_array = np.zeros((N_fft,Components))
 speech_sigmas_array = np.zeros((N_fft,Components))
 for No_of_data in range (0,len(directories)):
   samplerate, data = wavfile.read(path_to_VAD_speech+ directories[No_of_data])
   #print(len(data))
   data = data[int(len(data)/2):len(data)]
   #print(len(data))
   Bit_Check = wave.open(path_to_VAD_speech+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
  
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   for No_of_overlaps in range (0,Overlaps-10):
     #Rectangular_windowed_signal = data[int((Overlaps/2)*128)+128*No_of_overlaps:int((Overlaps/2)*128)+512+128*No_of_overlaps] #sliding window from approximately 1/2 of the signal
     Rectangular_windowed_signal = data[0+128*No_of_overlaps:512+128*No_of_overlaps]
     FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
     Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
     PSD_window_scaling = np.sum(Hann_window**2)
     PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)
     PSD_of_overlaps[:,No_of_overlaps] = PSD_of_windowed_signal


   for frequency_bin in range (0,N_fft):
       Mean_PSD_val[frequency_bin] = np.mean(PSD_of_overlaps[frequency_bin,:])
     
        
   for Frequency_bin in range (0,np.size(PSD_of_windowed_signal)):
      pre_codebook_array[Frequency_bin,No_of_data] = Mean_PSD_val[Frequency_bin]
   
 #print(pre_codebook_array)
 myFile = open('Mean_check.txt', 'r+')
 np.savetxt(myFile, pre_codebook_array)
 myFile.close()
 for Frequency_bin in range (0,N_fft):
    Frequency_bin_accross_all_data = pre_codebook_array[Frequency_bin,:] 
    Transpose = np.transpose(Frequency_bin_accross_all_data)
    n_iterations_pre_sieving = 5
    n_candidates = 100
    n_iterations_post_sieving = 100
    n_chosen_ones = 10
    
    class_probs, mean_vector, sigmas,log_likelihood = em_sieved( Transpose,
        Components,
        n_iterations_pre_sieving,
        n_candidates,
        n_iterations_post_sieving,
        n_chosen_ones,
        seed)
    speech_codebook_array[Frequency_bin,:] = mean_vector
    speech_CompProb_array[Frequency_bin,:] = class_probs
    speech_sigmas_array[Frequency_bin,:] = sigmas
    # for log_likelihood_history in log_likelihood:
    #     plt.plot(log_likelihood_history)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Log Likelihood")
    #     plt.savefig('Speech_FFT_Log_Likelihood/'+'Frequency_bin_'+str(Frequency_bin)+'.png')
    #     plt.close()

    print(mean_vector)
 return speech_CompProb_array, speech_codebook_array, speech_sigmas_array

def trainGMMnoise(Components,iterations,seed):
 directories = os.listdir(Path_to_noise)
 N_fft = 1024
 pre_codebook_array = np.zeros((N_fft,len(directories)))
 Noise_codebook_array = np.zeros((N_fft,Components))
 Noise_CompProb_array = np.zeros((N_fft,Components))
 Noise_sigmas_array = np.zeros((N_fft,Components))
 for No_of_data in range (0,len(directories)):
   samplerate, data = wavfile.read(Path_to_noise+'/'+ directories[No_of_data])
   Bit_Check = wave.open(Path_to_noise+'/'+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
  
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   for No_of_overlaps in range (0,Overlaps-5):
     Rectangular_windowed_signal = data[0+128*No_of_overlaps:512+128*No_of_overlaps]
     FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
     Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
     PSD_window_scaling = np.sum(Hann_window**2)
     PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)
     PSD_of_overlaps[:,No_of_overlaps] = PSD_of_windowed_signal


   for frequency_bin in range (0,N_fft):
       Mean_PSD_val[frequency_bin] = np.mean(PSD_of_overlaps[frequency_bin,:])
       #print(Mean_PSD_val[frequency_bin])
        
   for Frequency_bin in range (0,np.size(PSD_of_windowed_signal)):
      pre_codebook_array[Frequency_bin,No_of_data] = Mean_PSD_val[Frequency_bin]
      #print(PSD_of_windowed_signal[Frequency_bin])
 
 myFile = open('Noise_check.txt', 'r+')
 np.savetxt(myFile, pre_codebook_array)
 myFile.close()
 print('done!')
 for Frequency_bin in range (0,N_fft):
    Frequency_bin_accross_all_data = pre_codebook_array[Frequency_bin,:] 
    Transpose = np.transpose(Frequency_bin_accross_all_data)
    #print(np.shape(Transpose))
    n_iterations_pre_sieving = 5 #do the EM a little bit and check log likelihoods
    n_candidates = 100 #select 100 candidates
    n_iterations_post_sieving = 100  
    n_chosen_ones = 10
    
    class_probs, mean_vector, sigmas,log_likelihood = em_sieved( Transpose,
        Components,
        n_iterations_pre_sieving,
        n_candidates,
        n_iterations_post_sieving,
        n_chosen_ones,
        seed)
    Noise_codebook_array[Frequency_bin,:] = mean_vector
    Noise_CompProb_array[Frequency_bin,:] = class_probs
    Noise_sigmas_array[Frequency_bin,:] = sigmas
    # for log_likelihood_history in log_likelihood:
    #     plt.plot(log_likelihood_history)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Log Likelihood")
    #     plt.savefig('Noise_FFT_Log_likelihood/'+'Frequency_bin_'+str(Frequency_bin)+'_'+'Convergence.png')
    #     plt.close()
    print(mean_vector)
 return Noise_CompProb_array,Noise_codebook_array,Noise_sigmas_array

def generate_filtered_speech():
 text_file = open(Noise_file_for_filter, "r")  #make sure this is at the same location as this file
 lines = text_file.readlines()
 text_file.close()
 Noise_codebook2 = np.zeros((1024,Noise_clusters))

 speech_file = open("Speech_list.txt", "r") #make sure this is at the same location as this file
 speech_lines = speech_file.readlines()
 speech_file.close()
 Speech_codebook2 = np.zeros((1024,Speech_clusters))

 for frequency_bin in range (0,len(lines)):
  string_list = lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Noise_codebook2[frequency_bin,component] = 0
   else:
    Noise_codebook2[frequency_bin,component] = float(string_list[component])

 for frequency_bin in range (0,len(speech_lines)):
  string_list = speech_lines[frequency_bin].split()
  print(string_list)
  for component in range (0, len(string_list)):
   if string_list[component] == "nan":
    Speech_codebook2[frequency_bin,component] = 0
   else:
    Speech_codebook2[frequency_bin,component] = float(string_list[component])


 N_fft = 1024
 directories = os.listdir(Path_to_noisy_speech) #put the directory here. Could modify this for all the cases (-3 to -9dB) but for now lets test -6dB
 for No_of_data in range (0,20):
   samplerate, data = wavfile.read(Path_to_noisy_speech+'/'+ directories[No_of_data])
   Bit_Check = wave.open(Path_to_noisy_speech+'/'+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   audio= np.zeros(len(data))
   for No_of_overlaps in range (0,Overlaps-5): #need to fix this, ignores the last few parts
     
     Rectangular_windowed_signal = data[0+128*No_of_overlaps:512+128*No_of_overlaps]
     Estimated_speech_PSD = np.zeros(N_fft)
     Estimated_noise_PSD = np.zeros(N_fft)
     FFT_of_windowed_signal = Hann_window_a_signal(Rectangular_windowed_signal)
     Hann_window = sps.windows.hann(len(Rectangular_windowed_signal))
     PSD_window_scaling = np.sum(Hann_window**2)
     PSD_of_windowed_signal = (np.abs(FFT_of_windowed_signal)**2)/(samplerate*PSD_window_scaling)
     
     
     Noise_inverse = np.linalg.pinv(Noise_codebook2, rcond=1e-15)
     Noise_coeffs = Noise_inverse*PSD_of_windowed_signal
     Speech_inverse = np.linalg.pinv(Speech_codebook2, rcond=1e-15)
     Speech_coeffs = Speech_inverse*PSD_of_windowed_signal
     Speech_coeffs = np.transpose(Speech_coeffs)
     Noise_coeffs = np.transpose(Noise_coeffs)
     Estimated_speech_PSD_codebook = (Speech_coeffs*Speech_codebook2).clip(min=0)
     Estimated_noise_PSD_codebook = (Noise_coeffs*Noise_codebook2).clip(min=0)

     for Freq_bin in range (0,N_fft):
       Estimated_speech_PSD[Freq_bin]=np.sum(Estimated_speech_PSD_codebook[Freq_bin,:])
       Estimated_noise_PSD[Freq_bin]=np.sum(Estimated_noise_PSD_codebook[Freq_bin,:])

     Noise_suppression = 2 #These three value control the balance between noise suppression and the quality of the recovered speech
     Speech_emphasis = 1  
     Weiner_scaling = 1


     Current_frame_weiner_coeffs = Estimated_speech_PSD/(Noise_suppression*Estimated_noise_PSD+Speech_emphasis*Estimated_speech_PSD)
     De_noised_frame = (Current_frame_weiner_coeffs**Weiner_scaling)*FFT_of_windowed_signal
     FFT_to_audio = np.fft.ifft(De_noised_frame)
     audio[0+128*No_of_overlaps:512+128*No_of_overlaps] = audio[0+128*No_of_overlaps:512+128*No_of_overlaps]+FFT_to_audio[0:512] #recover only the windowed signal and not the zero-pad
   sf.write('Baseline/'+ directories[No_of_data], audio, 16000, 'PCM_16')     



Noise_clusters = 9 #Same as that of the research paper GMM codebook something something
Speech_clusters = 6
iterations = 1000
seed = 21

#Training takes hours
#comment this section out after training
Noise_CompProb_array, Noise_codebook, Noise_sigmas_array = trainGMMnoise(Noise_clusters,iterations,seed)
# myFile = open(Noise_file_for_filter, 'r+')
# np.savetxt(myFile, Noise_codebook)
# myFile.close()

# myFile = open('Noise_probs.txt', 'r+')
# np.savetxt(myFile, Noise_CompProb_array)
# myFile.close()

# myFile = open('Noise_sigma.txt', 'r+')
# np.savetxt(myFile, Noise_sigmas_array)
# myFile.close()



# speech_CompProb_array, Speech_codebook, speech_sigmas_array = trainGMMspeech(Speech_clusters,iterations,seed)
# myFile = open('Speech_list.txt', 'r+')
# np.savetxt(myFile, Speech_codebook)
# myFile.close()

# myFile = open('Speech_probs.txt', 'r+')
# np.savetxt(myFile, speech_CompProb_array)
# myFile.close()

# myFile = open('Speech_sigma.txt', 'r+')
# np.savetxt(myFile, speech_sigmas_array)
# myFile.close()


#comment this section after training


#Uncomment the line underneath if training is desired.
#generate_filtered_speech() 
#After training, we use the original non-sieving GMM for the adversarial network







     


