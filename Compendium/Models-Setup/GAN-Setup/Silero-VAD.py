import numpy as np
import os
import wave
from scipy.io import wavfile
import soundfile as sf
import torch
torch.set_num_threads(1)

#ATTENTION!
#This file applies the VAD_SNR definition.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

os.makedirs('Dataset/VAD_SNR',exist_ok=True)

Folder_path = 'Dataset/VAD_SNR'
Folder_path_save = 'Dataset/VAD_SNR/'
Path_Before_VAD = 'Dataset/dataset'
#Add lowr SNR like so: [9,6,3,0,-3,-6,-9,-12,-15,-18,-21,-24,-27,-30]
targets = [9,6,3,0,-3,-6,-9] 
#This creates the requires sub-directories.
#

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join(Folder_path,directory) 
 os.makedirs(path, exist_ok=True)

for folder in range (0,len(targets)):
 directory = str(targets[folder])+'dB'
 path = os.path.join(Folder_path+"/noise",directory) 
 os.makedirs(path, exist_ok=True) 


Speech_dir = os.listdir(Path_Before_VAD+'/clean')
Path_to_speech = Path_Before_VAD+'/clean'

SNR_check = np.zeros((len(Speech_dir),2))
Segment_length_in_seconds = 0.1
Sampling_period = 1 / 16000
N_samples_per_seg = int(Segment_length_in_seconds / Sampling_period)

for target_db in range (0,len(targets)):
  Noise_dir = os.listdir(Path_Before_VAD+'/noise/'+str(targets[target_db])+'dB')
  Path_to_noise = Path_Before_VAD+'/noise/'+str(targets[target_db])+'dB'
  Path_to_save = Folder_path_save+str(targets[target_db])+'dB'
  Path_to_save_for_noise = Folder_path_save+'noise/'+str(targets[target_db])+'dB'
  SNR_target = targets[target_db]
  for No_of_data in range (0,len(Speech_dir)):
   wav = read_audio(Path_to_speech+'/'+Speech_dir[No_of_data], sampling_rate=16000)
   samplerate, Speech_data = wavfile.read(Path_to_speech+"/"+ Speech_dir[No_of_data])
   Bit_Check = wave.open(Path_to_speech+"/"+ Speech_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Speech_data = Speech_data/(2**(bit_depth-1))


   samplerate, Noise_data = wavfile.read(Path_to_noise+'/'+Noise_dir[No_of_data])
   Bit_Check = wave.open(Path_to_noise+'/'+ Noise_dir[No_of_data], 'rb')
   bit_depth = Bit_Check.getsampwidth() * 8
   Noise_data = Noise_data/(2**(bit_depth-1))


   Noise = read_audio(Path_to_noise+'/'+Noise_dir[No_of_data], sampling_rate = 16000)
   speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
   

   #If the whole section had speech, then just apply the SNR definition.
   if not(len(speech_timestamps) == 0):
    Noise_timestamps = speech_timestamps
    Overlapping_section_of_noise = collect_chunks(Noise_timestamps,Noise)
    Speech_filtered_by_VAD = collect_chunks(speech_timestamps, wav)
    Speech_Numpy = Speech_filtered_by_VAD.numpy()
    Noise_Numpy = Overlapping_section_of_noise.numpy()
    Power_of_Speech = np.sum(Speech_Numpy ** 2)
    Power_of_Noise = np.sum(Noise_Numpy ** 2)
    Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))
    Noise_Numpy = Multiple * Noise_Numpy
    Power_of_Noise = np.sum(Noise_Numpy ** 2)
    snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)
    Adjusted_noisy_speech = Speech_data+Multiple*Noise_data
    sf.write(Path_to_save+'/'+str(Speech_dir[No_of_data]), Adjusted_noisy_speech, 16000, 'PCM_16')
    sf.write(Path_to_save_for_noise+'/'+str(Speech_dir[No_of_data]), Multiple*Noise_data, 16000, 'PCM_16')

   #Otherwise, find the overlapping sections, concatenate them and perform the SNR calculation
   #in order to boost the overlapping sections of noise.
   else:
   
    Speech_Numpy = wav.numpy()
    Noise_Numpy = Noise.numpy()
    Power_of_Speech = np.sum(Speech_Numpy ** 2)
    Power_of_Noise = np.sum(Noise_Numpy ** 2)

    Multiple = np.sqrt(Power_of_Speech / (Power_of_Noise * (10 ** (SNR_target / 10))))
    Noise_Numpy = Multiple * Noise_Numpy
    Power_of_Noise = np.sum(Noise_Numpy ** 2)
    snr_global = 10 * np.log10(Power_of_Speech / Power_of_Noise)
    Adjusted_noisy_speech = Speech_data+Multiple*Noise_data
    sf.write(Path_to_save+'/'+str(Speech_dir[No_of_data]), Adjusted_noisy_speech, 16000, 'PCM_16')
    sf.write(Path_to_save_for_noise+'/'+str(Speech_dir[No_of_data]), Multiple*Noise_data, 16000, 'PCM_16')
pass




