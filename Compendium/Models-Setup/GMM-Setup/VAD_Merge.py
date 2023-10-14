import os
import torch

torch.set_num_threads(1) #Change to the desired number of CPU threads to utilise.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')

os.makedirs('Models-Setup/GMM-Setup/VAD_merged_speech',exist_ok=True)
print(os. getcwd())
#This file eliminates the silent sections of speech and concatenates the portions.
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
Speech_dir = os.listdir('Dataset/dataset/clean')
Path_to_speech = 'Dataset/dataset/clean'
Path_to_save = 'Models-Setup/GMM-Setup/VAD_merged_speech'
for No_of_data in range (0,len(Speech_dir)):
 wav = read_audio(Path_to_speech+'/'+Speech_dir[No_of_data], sampling_rate=16000)
 speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
 if not(len(speech_timestamps) == 0):
  Speech_filtered_by_VAD = collect_chunks(speech_timestamps, wav)
  save_audio(Path_to_save+'/'+str(Speech_dir[No_of_data]),
    Speech_filtered_by_VAD, sampling_rate=16000)  
 else:
  save_audio(Path_to_save+'/'+str(Speech_dir[No_of_data]),
    wav, sampling_rate=16000)
  
 