function [ A, Means, Variances, pi ] = EM_estimate_gaussian(Y, Nhidden, ...
    Niter, epsilon)
%EM_ESTIMATE_GAUSSIAN EM algorithm for an HMM with Gaussian observations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of the parameters

% Initial transition matrix is random but stochastic (rows sum to 1)
% For uniform starting point: A = (1/Nhidden) * ones(Nhidden, Nhidden)
A = normalize(rand(Nhidden), 2);

% Initial means and variances of the emission probabilities

% Randomly sample points from the data so as to maximize their squared
% distances
Means = initMeansMaxCover(Y, Nhidden);

% Random between 0 and 1, should be an optional input parameter
Variances = rand(Nhidden, 1);

% Variances = initVariancesCheat(Y, Means, Nhidden);

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
    
    % Accumulators to sum over the sequences
    A1 = zeros(Nhidden, Nhidden);
    pi1 = zeros(Nhidden, 1);
    Means1 = zeros(Nhidden, 1);
    Variances1 = zeros(Nhidden, 1);
    SumGamma = zeros(Nhidden, 1);
    Gammas = zeros(Nhidden, T, N);
    
    LogLike_p = LogLike;
    LogLike = 0;
    
    for n=1:N
        if isnan(LogLike)
            break
        end
        
        X = Y(n,:);
        
        b = computeSmallB_Gaussian(X, Means, Variances, Nhidden, T);

        % Estimation
        [ alpha, beta, gamma, ll, ~ ] = ForwardBackwardSmoothing( A, b, ...
            pi, Nhidden, T );
        posterior = SmoothedPosterior(A, b, alpha, beta, T, Nhidden);

        LogLike = LogLike + ll;

        % Maximization: initial dist. of the hidden states
        pi1 = pi1 + gamma(:, 1) / sum(gamma(:, 1));
        
        % Maximization: transition matrix
        A1 = A1 + sum(posterior(:,:,2:end),3);
        
        % Maximization: means and variances
        Means1 = Means1 + sum( gamma .* repmat( X, Nhidden, 1 ) , 2 );
        
        % Store these results to re-use in the computation of the variance
        SumGamma = SumGamma + sum(gamma, 2);
        Gammas(:, :, n) = gamma;
    end
    Means1 = Means1 ./ SumGamma;
    
    if isnan(LogLike)
        fprintf('Log-likelihood is NaN - aborting!\n');
        break
    end
    
    for n=1:N
        % Estimation
        Cent = bsxfun(@minus, Y(n,:), Means1);
        Variances1 = Variances1 + sum( Gammas(:, :, n) .* (Cent .^ 2) , 2 );
    end
    
    % Normalize all accumulators
    A = normalize(A1, 2);
    pi = normalize(pi1);
    
    Means = Means1;
    Variances = Variances1 ./ SumGamma;
    
    LogLike = LogLike / N;
    i = i + 1;
end

end