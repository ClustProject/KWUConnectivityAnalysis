function DistEn = DistEn(sig, m, tau, B)
%DISTEN distribution entropy
% 
% Input parameters
% ------------------------------------------------
%       sig: Signal (time-series) under analysis
%         m: Embedding dimension
%       tau: Time delay
%         B: Number of histogram bins
% 
% Output parameters
% ------------------------------------------------
%    DistEn: Distribution entropy value
% 
% $Date:    17 Jun 2015
% $Modif.:  
% 

% parse inputs
narginchk(4, 4);

% rescaling


% distance matrix
N   = length(sig) - (m-1)*tau;
ind = hankel(1:N, N:length(sig));   
rnt = sig(ind(:, 1:tau:end));
dv  = pdist(rnt, 'chebychev');     

% esimating probability density by histogram
num  = hist(dv, linspace(0, 1, B));
freq = num./sum(num);
%freq = ePDF(dv,B);

% disten calculation
DistEn = -sum(freq.*log2(freq+eps)) ./ log2(B);
end
