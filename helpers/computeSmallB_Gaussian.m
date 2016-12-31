function [ b ] = computeSmallB_Gaussian(Y, Means, Variances, Nhidden, T)
% Vectorized computation of the probabilities is 10 times as fast as a for
% loop
X = repmat(Y, Nhidden, 1);
M = repmat(Means,1,T);
S = repmat(sqrt(Variances),1,T);

b = normpdf(X, M, S);
end
