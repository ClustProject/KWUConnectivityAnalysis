function wave = brain_waves(sig, fs, n)

%delta : 1-3Hz
%theta : 4-7Hz
%alpha : 8-13Hz
%Beta  : 14-30Hz
%Gamma : 31-50Hz
fc1_list = [1,4,8,14,31];
fc2_list = [3,7,13,30,50];
wave = [];
for i=1:5
    fc1 = fc1_list(i);
    fc2 = fc2_list(i);
    bpf = designfilt('bandpassiir', 'FilterOrder', n, ...
          'HalfPowerFrequency1', fc1, 'HalfPowerFrequency2', fc2, ...
          'SampleRate', fs);
    
    wave = [wave;filtfilt(bpf, sig)];
end
% Plot
%-----------------------------------------------------------
% hold;
% N = length(sig);
% tn = (0:N-1)/fs;
% plot(tn, sig);
% hold on;
% plot(tn, wave(1,:));
% plot(tn, wave(2,:));
% plot(tn, wave(3,:));
% plot(tn, wave(4,:));
% plot(tn, wave(5,:));
% 
% title 'Electroencephalogram';
% xlabel 'Time (s)';
% legend('Original','''Delta''','''Theta''', '''Alpha''', '''Beta''', '''Gamma''');
% grid;

end
%fvtool(bpf)

