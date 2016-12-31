function [ alpha, beta, gamma, lp, Z ] = ForwardBackwardSmoothing( A, b, ...
    pi, N, T )
%FORWARDBACKWARDSMOOTHING Smoothing using the forward-backward algorithm
%   Section 17.4.x of K. Murphy's book
    [alpha, lp, Z] = ForwardFiltering( A, b, pi, N, T );
    beta = BackwardFiltering(A, b, N, T);
    gamma = normalize(alpha .* beta, 1);
    
end

function [ beta ] = BackwardFiltering(A, b, N, T)
    beta = zeros(N, T);
    beta(:, T) = ones(N, 1);
    for t=T:-1:2
        beta(:,t-1) = normalize(A * (beta(: ,t) .* b(:,t)));
    end
end