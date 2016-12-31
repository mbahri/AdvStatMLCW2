function [ A, B, pi ] = EM_estimate_discrete(Y, Nhidden, Niter, ...
    epsilon)
%EM_ESTIMATE_DISCRETE EM algorithm for an HMM with discrete observations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

% X sparse coding
Nv = length(unique(Y));
X = zeros(T, Nv);
for i=1:T
    X(i, Y(i)) = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of the parameters

% Initial transition matrix is random but stochastic (rows sum to 1)
% For uniform starting point: A = (1/Nhidden) * ones(Nhidden, Nhidden)
A = normalize(rand(Nhidden), 2);

% Initial B is random, could be uniform of size Nv*Nhidden with a small
% perturbation.
% reg = 1e-2*rand(Nhidden, Nv);
% B = normalize( (1/Nv)*ones(Nhidden, Nv) + reg , 2);
B = normalize(rand(Nhidden, Nv), 2);

% Uniform: pi = (1/Nhidden) * ones(Nhidden, 1)
pi = normalize(rand(Nhidden,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM algorithm
i=0;
LogLike_p = -Inf;
LogLike = -Inf;

while i<Niter && (LogLike > LogLike_p + epsilon*abs(LogLike_p) || ...
        LogLike == -Inf || LogLike_p == -Inf)
    fprintf('[%02d] - Log-Likelihood: %f\n', i+1, LogLike);
    
    % Accumulators so sum over the sequences
    A1 = zeros(Nhidden, Nhidden);
    pi1 = zeros(Nhidden, 1);
    B1 = zeros(Nhidden, Nv);
    
    LogLike_p = LogLike;
    LogLike = 0;
    
    for n=1:N
        if isnan(LogLike)
            break
        end
        b = computeSmallB_Discrete(Y(n,:), B);

        % Estimation
        [ alpha, beta, gamma, ll, ~ ] = ForwardBackwardSmoothing( A, b, ...
            pi, Nhidden, T );
        posterior = SmoothedPosterior(A, b, alpha, beta, T, Nhidden);

        LogLike = LogLike + ll;
        
        % Maximization: initial dist. of the hidden states
        pi1 = pi1 + gamma(:, 1) / sum(gamma(:, 1));
        
        % Maximization: transition matrix
        A1 = A1 + sum(posterior(:,:,2:end),3);
        
        % Maximization: emission matrix
        B1 = B1 + gamma * X;
    end
    if isnan(LogLike)
        break
    end
    
    % Normalize all accumulators
    A = normalize(A1, 2);
    % Try adding a small perturbation to help avoid bad local extrema, not
    % very effective. Set to 0 to disable.
%     reg = 0;
    reg = rand(Nhidden, Nv);
    B = normalize(normalize(B1, 2) + reg, 2);
    pi = normalize(pi1);
    
    LogLike = LogLike / N;
    i = i + 1;
end

end
