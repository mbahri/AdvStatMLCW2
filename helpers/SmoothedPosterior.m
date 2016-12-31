function [ post ] = SmoothedPosterior(  A, b, alpha, beta, T, Nhidden  )
%SMOOTHEDPOSTERIOR Computes the two-slice smoothed posterior from alpha,
% beta, A and b.

post = zeros(Nhidden, Nhidden, T-1);

for t=1:T-1
    post(:, :, t) = normalize(A .* (alpha(:, t) * (b(:, t+1) .* beta(:, t+1))'));
end

