function dde = DDE(s)
time_min = size(s,2);
dde = zeros(time_min,1);

for j = 1:time_min
    v = var(s(1:end, j));
    dde(j) = log2(2*pi*exp(1)*v);
end

end