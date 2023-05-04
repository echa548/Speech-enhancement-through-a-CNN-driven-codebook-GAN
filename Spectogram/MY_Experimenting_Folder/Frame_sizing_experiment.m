clear
clc
Fs=48000;
Fs2=16000;
Sampling_Frequencies = [Fs,Fs2];
Frame_dur = 0.032;
Frame_size = Sampling_Frequencies(2)*Frame_dur;
Lowest_Frequency = 5*Sampling_Frequencies(2)/Frame_size;
Frequency_resolution = Sampling_Frequencies(2)/Frame_size;
Temporal_resolution = 1/Frequency_resolution;