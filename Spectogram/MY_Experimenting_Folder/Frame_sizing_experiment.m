clear
clc
Fs=48000;
Fs2=11025;
Sampling_Frequencies = [Fs,Fs2];
Frame_dur = 0.04;
Frame_size = Sampling_Frequencies(2)*Frame_dur;
Lowest_Frequency = 5*Sampling_Frequencies(2)/Frame_size;
Frequency_resolution = Sampling_Frequencies(2)/Frame_size;
Temporal_resolution = Frame_size/Sampling_Frequencies(2);