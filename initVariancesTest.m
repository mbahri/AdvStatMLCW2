function [ Variances ] = initVariancesTest( Y, Means, Nhidden )
%Test of a method for estimating the variances.

Variances = zeros(Nhidden, 1);
for i=1:Nhidden
    M = Means(i);
    Ninetyfive = (Y >= 0.5250*M & Y <= 1.4750*M);
    N = sum(Ninetyfive(:)) - 1;
    if N > 50
        V = var(Y(Ninetyfive));
        Variances(i) = N*V/chi2rnd(N);
    else
        Variances(i) = rand();
    end
end

end

