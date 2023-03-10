from scipy.io import wavfile
import math
import numpy as np
from scipy import signal
import scipy.signal as sps
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt

samplerate, data = wavfile.read("MY_Experimenting_Folder/Medium48k.wav")

Fs1 = samplerate
Fs2 = 11025
N = len(data)
total_time = (N-1)/Fs1
Max_Signal_Frequency =Fs2/2
New_sample_amount = math.ceil(Fs2*total_time)
Left_channel_Stereo = np.zeros(New_sample_amount)
Right_channel_Stereo = np.zeros(New_sample_amount)
data = data/(2**(24-1))
Left_channel = data[:,0]

Right_channel = data[:,1]

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

for i in range (0,2):  
 Original_signal = data[:,i]
 
 Anti_Aliased_signal = np.array(butter_lowpass_filter(Original_signal,Max_Signal_Frequency-1,Fs1))
 Down_sampled_signal = np.array(sps.resample(Anti_Aliased_signal,New_sample_amount))
 
 if i ==0:
  Left_channel_Stereo = Down_sampled_signal
 elif i==1:
    Right_channel_Stereo = Down_sampled_signal

print(np.shape(Left_channel_Stereo))
Left_channel_right_channel = np.vstack((Left_channel_Stereo, Right_channel_Stereo))

Left_channel_right_channel=Left_channel_right_channel.transpose()
#Only uncomment if a file needs to be downsampled
wavfile.write('MY_Experimenting_Folder/abc1.wav', Fs2, Left_channel_right_channel*169.84)

#Change directory to downsampled file of interest
Down_Sampled_rate, Downsampled_data = wavfile.read('MY_Experimenting_Folder/abc1.wav')
Downsampled_data = Downsampled_data
Down_Sampled_data_left = Downsampled_data[:,0]
Down_Sampled_data_right = Downsampled_data[:,1]

#Change directories as required
f, t, Lxx = signal.spectrogram(x = Down_Sampled_data_left, fs = Down_Sampled_rate, window = 'hann',nperseg = 512,noverlap = 384,nfft = 512,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
plt.pcolormesh(t, f, 10 * np.log10(Lxx), cmap ='magma')
plt.colorbar(label='Decibels')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('Spectrogram_Plots\\Magnitude_Plot_Left.png')
plt.close()

f, t, Rxx = signal.spectrogram(x = Down_Sampled_data_right,fs = Down_Sampled_rate,window = 'hann',nperseg = 512,noverlap = 384,nfft =  512,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
plt.pcolormesh(t, f, 10 * np.log10(Rxx), cmap ='magma')
plt.colorbar(label='Decibels')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('Spectrogram_Plots\\Magnitude_Plot_Right.png')
plt.close()
Phase, t_phase, Lxx_Phase = signal.spectrogram( x = Down_Sampled_data_left, fs = Down_Sampled_rate, window = 'hann', nperseg = 512,noverlap = 384,nfft = 512,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
plt.pcolormesh(t_phase, Phase, Lxx_Phase, cmap ='magma')
plt.colorbar(label='Phase')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('Spectrogram_Plots\\Phase_Plot_Left.png')
plt.close()

Phase, t_phase, Rxx_Phase = signal.spectrogram(x = Down_Sampled_data_right,fs = Down_Sampled_rate, window = 'hann', nperseg = 512, noverlap = 384, nfft = 512,detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
plt.pcolormesh(t_phase, Phase, Rxx_Phase, cmap ='magma')
plt.colorbar(label='Phase')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('Spectrogram_Plots\\Phase_Plot_Right.png')
plt.close()



#Keep here for now, this is just for prototyping purposes and also if we want to switch to librosa
#Left_channel_STFT = librosa.stft(y = Down_Sampled_data_left/(2**(24-1)),n_fft = 512,hop_length = 128, win_length =512, window ='hann', center = True, dtype = None)  
#Right_channel_STFT = librosa.stft(y = Down_Sampled_data_right/(2**(24-1)),n_fft = 512,hop_length = 128, win_length =512, window ='hann', center = True, dtype = None)  
#ISTFT 
#phase_Left = np.angle(Left_channel_STFT)
#phase_Right = np.angle(Left_channel_STFT)
#print(np.shape(Lxx))
#print(np.shape(Lxx_Phase))

#Change Lxx and Rxx to expected Wave-U-net Output

combined_Left = np.multiply(np.abs(Lxx), np.exp(1j * Lxx_Phase))
combined_Right = np.multiply(np.abs(Rxx), np.exp(1j * Rxx_Phase))
#print(np.shape(combined_Left))
Real_signal_Left = np.array(sps.istft(Zxx = combined_Left,fs = Fs2,  window = 'hann',nperseg = 512,noverlap = 384,nfft = 512,input_onesided=True,boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'))
Real_signal_Right = np.array(sps.istft(Zxx = combined_Right,fs =Fs2,  window = 'hann',nperseg = 512,noverlap = 384,nfft = 512,input_onesided=True,boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'))

print(np.shape(Real_signal_Left))
#print(np.shape(Real_signal_Left))
#print (Real_signal_Left[1,:])

#ISTFT Algo, This sounds distorted try Librosa ISTFT instead of griffinlim
#Real_signal_Left = librosa.griffinlim(np.abs(Lxx))
#Real_signal_Right = librosa.griffinlim(np.abs(Rxx))
#Real_signal_Left = Real_signal_Left[2,:]
#print(Real_signal_Left)

#It finally fking works
Real_signal_Stereo= np.vstack((Real_signal_Left[1,:], Real_signal_Right[1,:]))
Real_signal_Stereo=Real_signal_Stereo.transpose()
wavfile.write('MY_Experimenting_Folder/Test_ABC.wav', Fs2, Real_signal_Stereo*5.7)