function [ S ] = ViterbiDecode( Y, Nhidden, type )
%VITERBIDECODE Perform Viterbi decoding on the smoothed data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM and setting the way to compute the vector b according to the type

if strcmp(type, 'gauss')
    [A, Mu, Sigma, Pi] = EM_estimate_gaussian(Y, Nhidden, 100, 1e-4);
    SmallB = @(X) computeSmallB_Gaussian(X, Mu, Sigma, Nhidden, length(X));
elseif strcmp(type, 'multinoulli')
    [A, B, Pi] = EM_estimate_discrete(Y, Nhidden, 100, 1e-4);
    SmallB = @(X) computeSmallB_Discrete(Y, B);
else
    error 'Invalid type: must be either gauss or multinoulli'
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Viterbi algorithm

S = zeros(N,T);

for n=1:N
    X = Y(n,:);
    b = SmallB(X);

    % If we really want to divide by c_t = Z(t)...
%     [ alpha, beta, gamma, ~, Z ] = ForwardBackwardSmoothing( A, b, ...
%         Pi, Nhidden, T );

    delta = zeros(Nhidden, T);
    a = zeros(Nhidden, T);

    % First element in delta: product of b_1 and pi
    delta(:,1) = log(b(:,1)) + log(Pi);

    % Compute delta and a for each time step
    for t=2:T
        temp = log(A) + repmat(delta(:,t-1),1,Nhidden);
        [max_temp, a(:,t)] = max(temp);
        delta(:, t) = log(b(:,t)) + max_temp';% - log(Z(t));
    end
    
    % The 'a' matrix has one step fewer than the observations because it
    % gives information on what was the most probable state before the
    % current state. So we initialize the last state by taking the most
    % likely according to delta, and we go backwards.
    [~, last] = max(delta(:,end));
    S(n,end) = last;
    for t=T:-1:2
        S(n,t-1) = a(S(n,t),t);
    end
end

end