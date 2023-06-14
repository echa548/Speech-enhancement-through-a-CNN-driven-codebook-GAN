from scipy.io import wavfile
import math
import numpy as np
from scipy import signal
from pathlib import Path
import scipy.signal as sps
from scipy.signal import butter, lfilter
import soundfile as sf
import matplotlib.pyplot as plt
import pydub
import uuid
import os
from pydub import AudioSegment, effects
import wave
#Dependencies on ffmpeg, install if u dont have. Add to PATH (Environment variables)
#AudioSegment.converter = "C:Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram/FFMPEG/ffmpeg.exe"
#AudioSegment.ffmpeg = "C:Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram/FFMPEG/ffmpeg.exe"
#AudioSegment.ffprobe ="C:Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram/FFMPEG/ffprobe.exe"
os.chdir('C:/Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700')
#print("Current Working Directory " , os.getcwd())

#data gathering and spectrogram generation
#external sources, only run once. Change number of Generate_data to randomize file names and write to a different directory. Change data directory as required.

Generate_data = 0
Generate_normalised_downsampled_speech = 1
Generate_normalised_downsampled_noise = 0

if Generate_normalised_downsampled_speech ==1:
 directories = os.listdir('Model/Convert_audio_to_spectrogram/audio/speech')
if Generate_normalised_downsampled_noise ==1:
 directories = os.listdir('Model/Convert_audio_to_spectrogram/audio/noise')  

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Generate_Mono_Spectrogram (wow):

  pass

if Generate_data == 1:
  for root, dirs, files in os.walk("C:\\Users\\Timothy\Desktop\\MY_Experimenting_Folder\\Test_Audio", topdown=True):
   for name in files:
      if (os.path.join(root, name).endswith('.wav')):
       samplerate, data = wavfile.read(os.path.join(root, name))
       sf.write('C:\\Users\\Timothy\\Desktop\\Random_data\\'+ str(uuid.uuid4())+'.wav', data, samplerate, 'PCM_16')
   for name in dirs:
      if (os.path.join(root, name).endswith('.wav')):
       samplerate, data = wavfile.read(os.path.join(root, name))
       sf.write('C:\\Users\\Timothy\\Desktop\\Random_data\\'+ str(uuid.uuid4())+'.wav', data, samplerate, 'PCM_16')
      

#print(directories)
for i in range (0,len(directories)):
   
   if Generate_normalised_downsampled_speech ==1:
    samplerate, data = wavfile.read("Model/Convert_audio_to_spectrogram/audio/speech/"+ directories[i])
    print(samplerate)
    Bit_Check = wave.open("Model/Convert_audio_to_spectrogram/audio/speech/"+ directories[i], 'rb')
   if Generate_normalised_downsampled_noise == 1:
    samplerate, data = wavfile.read("Model/Convert_audio_to_spectrogram/audio/noise/"+ directories[i])
    Bit_Check = wave.open("Model/Convert_audio_to_spectrogram/audio/noise/"+ directories[i], 'rb')


   Fs1 = samplerate
   Fs2 = 16000
   bit_depth = Bit_Check.getsampwidth() * 8
   
  #If already at 16Khz, just let it pass.
   if Fs2==Fs1:
     data = data/(2**(bit_depth-1))
     #signal inspection
     if Generate_normalised_downsampled_speech ==1:
      sf.write("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_speech/"+ directories[i], data, samplerate, 'PCM_16')
     if Generate_normalised_downsampled_noise == 1:
      sf.write("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_noise/"+ directories[i], data, samplerate, 'PCM_16')
       
     #TO be changed to local SNR instead of global RMS for a more accurate estimation.
     #This means adjusting the noise level to the local samples signal.
     #will need to detect if power of speech signal represents silence or not.



    #rawsound = AudioSegment.from_wav('MY_Experimenting_Folder/Processed_audio/'+directories[i])  
    #normalizedsound = effects.normalize(rawsound)  
    #normalizedsound.export('MY_Experimenting_Folder/Processed_audio/'+directories[i], format = 'wav')

#To do: Implement spectrogram function, sort out local , Ytb premium sub as well change to student
#Spectrogram generation (Mono) Uncomment when needed
    #  f, t, Lxx = signal.spectrogram(x = data, fs = samplerate, window = 'hann',nperseg = 640,noverlap = 480,nfft = 1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
    #  plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
    #  plt.colorbar(label='Decibels')
    #  plt.ylabel('Frequency [Hz]')
    #  plt.xlabel('Time [sec]')
    #  plt.savefig('Spectrogram_Plots/Mag/' + directories[i] + 'Magnitude_Plot_Mono.png')
    #  plt.close()
    #  Phase, t_phase, Lxx_Phase = signal.spectrogram( x = data, fs = samplerate, window = 'hann', nperseg = 640,noverlap = 480,nfft =  1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
    #  plt.pcolormesh(t_phase, Phase, Lxx_Phase, cmap ='magma')
    #  plt.colorbar(label='Phase')
    #  plt.ylabel('Frequency [Hz]')
    #  plt.xlabel('Time [sec]')
    #  plt.savefig('Spectrogram_Plots/Phase/' + directories[i] + 'Phase_Plot_Mono.png')
    #  plt.close()
   else:
     Mono_Stereo = np.shape(data)
     if len(Mono_Stereo)==1:
       N = len(data)
       total_time = (N-1)/Fs1
       Max_Signal_Frequency =Fs2/2
       New_sample_amount = math.ceil(Fs2*total_time)
       Single_Channel = np.zeros(New_sample_amount)
       data = data/(2**(bit_depth-1))
       Original_signal = data
       Anti_Aliased_signal = np.array(butter_lowpass_filter(Original_signal,Max_Signal_Frequency,Fs1))
       Down_sampled_signal = np.array(sps.resample(Anti_Aliased_signal,New_sample_amount))
       Single_Channel = Down_sampled_signal
       Transformed_single_channel=Single_Channel.transpose()
       #wavfile.write('Spectogram\\MY_Experimenting_Folder\\Processed_audio\\'+directories[i], Fs2, Transformed_single_channel)
       if Generate_normalised_downsampled_speech == 1:
        sf.write("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_speech/"+ directories[i], Transformed_single_channel, Fs2, 'PCM_16')
       if Generate_normalised_downsampled_noise == 1:
        sf.write("Spectogram/MY_Experimenting_Folder/Processed_audio/processed_noise/"+ directories[i], Transformed_single_channel, Fs2, 'PCM_16')
      #  rawsound = AudioSegment.from_wav('Spectogram\\MY_Experimenting_Folder\\Processed_audio\\'+directories[i])  
      #  normalizedsound = effects.normalize(rawsound)  
      #  normalizedsound.export('Spectogram\\MY_Experimenting_Folder\\Processed_audio\\'+directories[i], format = 'wav')
       #Down_Sampled_rate, Downsampled_data = wavfile.read('Spectogram\\MY_Experimenting_Folder\\Processed_audio\\'+directories[i])
       
      #  f, t, Lxx = signal.spectrogram(x = Downsampled_data, fs = Down_Sampled_rate, window = 'hann',nperseg = 640,noverlap = 480,nfft = 1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
      #  plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
      #  plt.colorbar(label='Decibels')
      #  plt.ylabel('Frequency [Hz]')
      #  plt.xlabel('Time [sec]')
      #  plt.savefig('Spectrogram_Plots/Mag/' + directories[i] + 'Magnitude_Plot_Mono.png')
      #  plt.close()
      #  Phase, t_phase, Lxx_Phase = signal.spectrogram( x = Downsampled_data, fs = Down_Sampled_rate, window = 'hann', nperseg = 640,noverlap = 480,nfft =  1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
      #  plt.pcolormesh(t_phase, Phase, Lxx_Phase, cmap ='magma')
      #  plt.colorbar(label='Phase')
      #  plt.ylabel('Frequency [Hz]')
      #  plt.xlabel('Time [sec]')
      #  plt.savefig('Spectrogram_Plots/Phase/' + directories[i] + 'Phase_Plot_Mono.png')
      #  plt.close()
     else:
      N = len(data)
      total_time = (N-1)/Fs1
      Max_Signal_Frequency =Fs2/2
      New_sample_amount = math.ceil(Fs2*total_time)
      Left_channel_Stereo = np.zeros(New_sample_amount)
      Right_channel_Stereo = np.zeros(New_sample_amount)
      data = data/(2**(32-1))
      Left_channel = data[:,0]
      Right_channel = data[:,1]

      for i in range (0,2):
       Original_signal = data[:,i] 
       Anti_Aliased_signal = np.array(butter_lowpass_filter(Original_signal,Max_Signal_Frequency,Fs1))
       Down_sampled_signal = np.array(sps.resample(Anti_Aliased_signal,New_sample_amount))
       if i ==0:
        Left_channel_Stereo = Down_sampled_signal
       elif i==1:
        Right_channel_Stereo = Down_sampled_signal
      Left_channel_right_channel = np.vstack((Left_channel_Stereo, Right_channel_Stereo))
      Left_channel_right_channel=Left_channel_right_channel.transpose()
      sf.write('MY_Experimenting_Folder\\Processed_audio\\'+directories[i], Left_channel_right_channel, Fs2, 'PCM_16')
      rawsound = AudioSegment.from_wav('MY_Experimenting_Folder\\Processed_audio\\'+directories[i])  
      normalizedsound = effects.normalize(rawsound)  
      normalizedsound.export('MY_Experimenting_Folder\\Processed_audio\\'+directories[i], format = 'wav')
      Down_Sampled_rate, Downsampled_data = wavfile.read('MY_Experimenting_Folder\\Processed_audio\\'+directories[i])
      Downsampled_data = Downsampled_data
      Down_Sampled_data_left = Left_channel_right_channel[:,0]
      Down_Sampled_data_right = Left_channel_right_channel[:,1]
      f, t, Lxx = signal.spectrogram(x = Down_Sampled_data_left, fs = Fs2, window = 'hann',nperseg = 640,noverlap = 480,nfft = 1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
      plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
      plt.colorbar(label='Decibels')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.savefig('Spectrogram_Plots/Mag/' + directories[i] + 'Magnitude_Plot_Left.png')
      plt.close()
      f, t, Rxx = signal.spectrogram(x = Down_Sampled_data_right,fs = Fs2, window = 'hann',nperseg = 512,noverlap = 384,nfft =  1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
      plt.pcolormesh(t, f, 10 * np.log10(Rxx), cmap ='Greys')
      plt.colorbar(label='Decibels')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.savefig('Spectrogram_Plots/Mag/' + directories[i] + 'Magnitude_Plot_Right.png')
      Phase, t_phase, Lxx_Phase = signal.spectrogram( x = Down_Sampled_data_left, fs = Fs2, window = 'hann', nperseg = 640,noverlap = 480,nfft =  1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
      plt.pcolormesh(t_phase, Phase, Lxx_Phase, cmap ='magma')
      plt.colorbar(label='Phase')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.savefig('Spectrogram_Plots/Phase/' + directories[i] + 'Phase_Plot_Left.png')
      plt.close()
      Phase, t_phase, Rxx_Phase = signal.spectrogram(x = Down_Sampled_data_right,fs = Fs2, window = 'hann', nperseg = 640,noverlap = 480,nfft =  1024,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
      plt.pcolormesh(t_phase, Phase, Rxx_Phase, cmap ='magma')
      plt.colorbar(label='Phase')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.savefig('Spectrogram_Plots/Phase/' + directories[i] + 'Phase_Plot_Right.png')
      plt.close()
#print(np.shape(Left_channel_Stereo))
#Left_channel_Stereo = effects.normalize(Left_channel_Stereo)
#Right_channel_Stereo = effects.normalize(Right_channel_Stereo)



#normalized_audio = effects.normalize(Left_channel_right_channel)
#Only uncomment if a file needs to be downsampled
#wavfile.write('C:/Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram/MY_Experimenting_Folder/abc1.wav', Fs2, Left_channel_right_channel)







#Change directory to downsampled file of interest


#Change directories as required











#Keep here for now, this is just for prototyping purposes and also if we want to switch to librosa
#Left_channel_STFT = librosa.stft(y = Down_Sampled_data_left/(2**(24-1)),n_fft = 512,hop_length = 128, win_length =512, window ='hann', center = True, dtype = None)  
#Right_channel_STFT = librosa.stft(y = Down_Sampled_data_right/(2**(24-1)),n_fft = 512,hop_length = 128, win_length =512, window ='hann', center = True, dtype = None)  
#ISTFT 
#phase_Left = np.angle(Left_channel_STFT)
#phase_Right = np.angle(Left_channel_STFT)
#print(np.shape(Lxx))
#print(np.shape(Lxx_Phase))

#Change Lxx and Rxx to expected Wave-U-net Output

#combined_Left = np.multiply(np.abs(Lxx), np.exp(1j * Lxx_Phase))
#combined_Right = np.multiply(np.abs(Rxx), np.exp(1j * Rxx_Phase))
#print(np.shape(combined_Left))
#Real_signal_Left = np.array(sps.istft(Zxx = combined_Left,fs = Fs2,  window = 'hann',nperseg = 640,noverlap = 480,nfft =  2048,input_onesided=True,boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'))
#Real_signal_Right = np.array(sps.istft(Zxx = combined_Right,fs =Fs2,  window = 'hann',nperseg = 640,noverlap = 480,nfft =  2048,input_onesided=True,boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'))

#print(np.shape(Real_signal_Left))
#print(np.shape(Real_signal_Left))
#print (Real_signal_Left[1,:])

#ISTFT Algo, This sounds distorted try Librosa ISTFT instead of griffinlim
#Real_signal_Left = librosa.griffinlim(np.abs(Lxx))
#Real_signal_Right = librosa.griffinlim(np.abs(Rxx))
#Real_signal_Left = Real_signal_Left[2,:]
#print(Real_signal_Left)

#It finally fking works
#Real_signal_Stereo= np.vstack((Real_signal_Left[1,:], Real_signal_Right[1,:]))
#Real_signal_Stereo=Real_signal_Stereo.transpose()
#sf.write('MY_Experimenting_Folder/Test_ABC.wav', Left_channel_right_channel, Fs2, 'PCM_24')
#rawsound = AudioSegment.from_wav("MY_Experimenting_Folder/Test_ABC.wav")  
#normalizedsound = effects.normalize(rawsound)  
#normalizedsound.export('MY_Experimenting_Folder/Test_ABC.wav', format = 'wav')
