clear
clc

[x, Fs] = audioread('Medium48k.wav');
Example = x;
Fs2=11025;
N = length(x);
total_time = (N-1)/Fs;
Max_signal_Frequency = Fs2/2;
New_sample_amount = ceil(Fs2*total_time);
 Stereo = zeros(New_sample_amount,2);
for i = 1:2
Original_signal = x(:,i);
Anti_Aliased_signal = lowpass (Original_signal,Max_signal_Frequency,Fs,'ImpulseResponse','iir');
N = length(Original_signal);
%  n=Fs2;
 T_sample = 1/Fs;
 Time_steps = 0:T_sample:total_time;
 [Down_sampled_signal,Time_resampled] = resample(Anti_Aliased_signal,Time_steps,Fs2);
  Stereo(:,i) = Down_sampled_signal;
end
Left_Channel_Right_Channel = [Stereo(:,1),Stereo(:,2)];
 audiowrite("MatLab_resample_Medium48k.wav",Left_Channel_Right_Channel,Fs2)
% sound(Stereo,Fs2);

%  Down_sampled_signal = lowpass (Down_sampled_signal,Max_signal_Frequency,Max_signal_Frequency,'ImpulseResponse','iir');



% Frame_dur = 0.1;
% Frame_size = Fs2*Frame_dur;
% Frequency_resolution = Fs/Frame_size;
% Temporal_resolution = Frame_size/Fs;
 
 % plot(total_time, x)

