function psd = PSD(s, f,num_bands,bandwidth)
time_min = size(s,2);
psd = zeros(time_min, num_bands);

for i = 1 : num_bands
    band = squeeze(bandwidth(:,i));
    fc1 = find(f>=band(1),1);
    fc2 = find(f>=band(2),1);
    for j = 1:time_min
        psd(j,i) = real(sum(abs(s(fc1:fc2,j).^2)))/(fc2-fc1+1);
    end
end

end