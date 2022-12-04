function de = DE(s,f,num_bands,bandwidth)
time_min = size(s,2);
de = zeros(time_min, num_bands);

for i = 1 : num_bands
        band = squeeze(bandwidth(:,i));
        fc1 = find(f>=band(1),1);
        fc2 = find(f>=band(2),1);
    for j = 1:time_min
        v = var(s(fc1:fc2, j));
        de(j,i) = log2(2*pi*exp(1)*v);
    end
end

end