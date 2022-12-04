imf = emd(djc_eeg1(1,:), 'MaxNumIMF', 10);

for i = 1 : 10
    opts = {'Window',hann(200,'periodic'),'FrequencyRange', "onesided",'FFTLength',512, 'OverlapLength', 0};
    [s,f] = stft(imf(:,i),200,opts{:});
    v = var(s(:,10));
    log2(2*pi*exp(1)*v)
end

