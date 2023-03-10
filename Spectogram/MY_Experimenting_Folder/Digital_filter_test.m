clear
clc

[x, Fs] = audioread('Easy_Sample.wav');

x= x(:, 1);
 N = length(x);
y = fft(x,N);
% f = (0:length(y)-1)*Fs/length(y);
NFFT = 2^nextpow2(N);
f = Fs/2*linspace(0,1,NFFT/2+1);

Magnitudes = abs(fft(x,NFFT));
% plot(Magnitudes)
plot (f,Magnitudes(1:NFFT/2+1))