function [ alpha, logp, Z ] = ForwardFiltering( A, b, pi, N, T )
%FORWARDFILTERING Filtering using the forward algorithm
%   Section 17.4.2 of K. Murphy's book
    Z = zeros(1, T);
    alpha = zeros(N, T);
    
    [alpha(:, 1), Z(1)] = normalize(b(:, 1) .* pi);
    for t=2:T
        [alpha(:, t), Z(t)] = normalize(b(:, t) .* (A' * alpha(:, t-1)));
    end
    logp = sum(log(Z));
end

