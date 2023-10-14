import numpy as np
import tensorflow_probability as tfp
import os
import wave
from scipy.io import wavfile
import math
import scipy.signal as sps
import soundfile as sf
import os
import wave
from scipy.special import logsumexp
#ATTENTION!
#This file implements a sieved univariate Gaussian Mixture Model (GMM) per frequency bin. This does not rely on any guesses on initial parameters.
#and assists with convergence (Provided it can be approximated through a sum of Gaussians).
#After training, the means of the GMM are saved in a TXT file for usage with a GAN model.

os.chdir('..')
samplerate =0
Path_to_noise = 'Dataset/dataset/noisy'


#Leave the lines underneath uncommented, this just lets the file compile.
Noise_file_for_filter = 'Noise_GMM_codebook.txt'
Speech_file_for_filter = 'Speech_GMM_codebook.txt'
#Leave the line underneath uncommented, this just lets the file compile.
Path_to_noisy_speech = 'Dataset/dataset/-6dB'
# Path_to_noisy_speech = 'creating_dataset2/dataset_all4/-6dB' #Original implementation.


def em(data, components, iterations, RNGseed):
 Number_of_samples = data.shape[0]
 np.random.seed(RNGseed)
 mean = np.random.rand(components)
 variance = np.random.rand(components)
 components_probs = np.random.dirichlet(np.ones(components))
 
 for em in range(iterations): 
 
  responsibilities=tfp.distributions.Normal(loc=mean, scale = variance).prob(data.reshape(-1,1)).numpy()*components_probs 

  responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True) 
  components_responsibilities = np.sum(responsibilities, axis=0) 
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
 directories = os.listdir('Models-Setup/GMM-Setup/Proper_speech')
 path_to_VAD_speech = 'Models-Setup/GMM-Setup/Proper_speech/'
 N_fft = 1024
 pre_codebook_array = np.zeros((N_fft,len(directories)))
 speech_codebook_array = np.zeros((N_fft,Components))
 speech_CompProb_array = np.zeros((N_fft,Components))
 speech_sigmas_array = np.zeros((N_fft,Components))
 for No_of_data in range (0,len(directories)):
   samplerate, data = wavfile.read(path_to_VAD_speech+ directories[No_of_data])
   data = data[int(len(data)/2):len(data)]
   Bit_Check = wave.open(path_to_VAD_speech+ directories[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   data = data/(2**(bit_depth-1))
  
   Overlaps = math.floor(len(data)/128)
   PSD_of_overlaps = np.zeros((N_fft,Overlaps))
   Mean_PSD_val = np.zeros(N_fft)
   for No_of_overlaps in range (0,Overlaps-10):
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
   
 for Frequency_bin in range (0,N_fft):
    Frequency_bin_accross_all_data = pre_codebook_array[Frequency_bin,:] 
    Transpose = np.transpose(Frequency_bin_accross_all_data)
    #This is how many times EM needs to run for each guess of initial parameters. Change as needed.
    n_iterations_pre_sieving = 5
    #This is how many initial guesses of the GMMs there are. Change as needed.
    n_candidates = 100
    #This is how many EM iterations will be done after selecting n_chosen_ones out of n_candidates that have the highest likelihood. Change as needed.
    n_iterations_post_sieving = 100
    #This is how many of the initialized models will be chosen. These models are the N highest likelihood models. Change as needed.
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
        
   for Frequency_bin in range (0,np.size(PSD_of_windowed_signal)):
      pre_codebook_array[Frequency_bin,No_of_data] = Mean_PSD_val[Frequency_bin]
 
 myFile = open('Noise_check.txt', 'r+')
 np.savetxt(myFile, pre_codebook_array)
 myFile.close()
 print('done!')

 for Frequency_bin in range (0,N_fft):
    Frequency_bin_accross_all_data = pre_codebook_array[Frequency_bin,:] 
    Transpose = np.transpose(Frequency_bin_accross_all_data)
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
    print(mean_vector)
 return Noise_CompProb_array,Noise_codebook_array,Noise_sigmas_array


#This is an improper function, This does not use normalised PSDs. #Do not use this function.
def generate_filtered_speech():
 text_file = open(Noise_file_for_filter, "r")  #make sure this is at the same location as this file
 lines = text_file.readlines()
 text_file.close()
 Noise_codebook2 = np.zeros((1024,Noise_clusters))

 #speech_file = open("Speech_list.txt", "r") #make sure this is at the same location as this file
 speech_file = open("Modified_speech_list.txt", "r")
 
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


Noise_clusters = 9 #Same as that of the research paper GMM codebook
Speech_clusters = 6
iterations = 1000
seed = 21

#These save to the same path as that of the GAN-Setup as one of the provided models makes use of the GMM derived codebook.
Noise_codebook_file_name_and_path = 'Models-Setup/GAN-Setup/Noise_GMM_codebook.txt' #These files contains the GMM means.
Speech_codebook_file_name_and_path = 'Models-Setup/GAN-Setup/Speech_GMM_codebook.txt'

Noise_codebook_GMM_Component_probabilities_file_name_and_path = 'Models-Setup/GAN-Setup/Noise_GMM_Component_probabilities.txt'
Speech_codebook_GMM_Component_probabilities_file_name_and_path = 'Models-Setup/GAN-Setup/Speech_GMM_Component_probabilities.txt'

Noise_codebook_GMM_Component_variance_file_name_and_path = 'Models-Setup/GAN-Setup/Noise_GMM_Component_variance.txt'
Speech_codebook_GMM_Component_variance_file_name_and_path = 'Models-Setup/GAN-Setup/Speech_GMM_Component_variance.txt'

Noise_CompProb_array, Noise_codebook, Noise_sigmas_array = trainGMMnoise(Noise_clusters,iterations,seed)
myFile = open(Noise_codebook_file_name_and_path, 'r+')
np.savetxt(myFile, Noise_codebook)
myFile.close()

#Uncomment these lines if the GMMs need to be assembled and graphed using the Plot_GMMs file.
#If running the Plot_GMMs script, then please change the directories of the saved codebooks to 'Models-Setup/GMM-Setup/Noise_GMM_codebook.txt' and similar for the other files.



# myFile = open(Noise_codebook_GMM_Component_probabilities_file_name_and_path, 'r+')
# np.savetxt(myFile, Noise_CompProb_array)
# myFile.close()

# myFile = open(Noise_codebook_GMM_Component_variance_file_name_and_path, 'r+')
# np.savetxt(myFile, Noise_sigmas_array)
# myFile.close()



speech_CompProb_array, Speech_codebook, speech_sigmas_array = trainGMMspeech(Speech_clusters,iterations,seed)
myFile = open(Speech_codebook_file_name_and_path, 'r+')
np.savetxt(myFile, Speech_codebook)
myFile.close()

# myFile = open(Speech_codebook_GMM_Component_probabilities_file_name_and_path, 'r+')
# np.savetxt(myFile, speech_CompProb_array)
# myFile.close()

# myFile = open(Speech_codebook_GMM_Component_variance_file_name_and_path, 'r+')
# np.savetxt(myFile, speech_sigmas_array)
# myFile.close()




#Do not use this function
#generate_filtered_speech() 








     


