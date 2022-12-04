function mde = MDE(sig)
energy_value = sum(power(sig,2));
second_term = 2.0*pi*exp(1)/200.0;
mean_sig = mean(sig);
mde = mean_sig*log(energy_value*second_term)/2.0;
%de = log(energy_value)/2.0 + log(second_term)/2.0;
%de = log(2.0*pi*exp(1)*var(sig))/2.0;

end