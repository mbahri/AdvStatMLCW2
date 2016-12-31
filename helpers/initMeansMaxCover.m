function [Means] = initMeansMaxCover(Y, Nhidden)
% Finds Nhidden means by randomly sampling the data such that they are
% maximally spread.
% See Kevin Murphy's book chapter 11 p357.
X = Y(:)';
Means = zeros(Nhidden, 1);

Means(1) = randsample(X, 1);
k = 1;

for i=2:Nhidden
    Delta = min(bsxfun(@minus, repmat(X, k, 1), Means(1:k)) .^ 2, [], 1);
    Means(i) = randsample(X, 1, true, Delta);
    k = k+1;
end

end