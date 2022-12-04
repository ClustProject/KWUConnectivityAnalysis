function xx = kalmanfilter(sig)
    %R = 10; %, cov = 0.000001
    %R = 1; %, cov = 0.01
    %DE_STFT_LDS --> R = 50, cov = 0.000001

    R = 1;
    w = 1;
    kalman = dsp.KalmanFilter('ProcessNoiseCovariance', 0.001,...
    'MeasurementNoiseCovariance',R,...
    'InitialStateEstimate',mean(sig(1:20)),...
    'InitialErrorCovarianceEstimate',1,...
    'ControlInputPort',false);
    sig_size = size(sig,2);
    xx = zeros(1, sig_size);
    for i = 1:sig_size
        x = sig(i);
        x_hat = kalman(x);
        xx(i) = x_hat;
%         w = w + 1;
    end
%     hold;
%     plot(sig);
%     hold on
%     plot(xx);
%     
%     legend('original', '''smoothed''');

end