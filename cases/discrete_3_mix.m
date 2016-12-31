%Example with three discrete distributions

N  = 1000;         % number of sequences
T  = 100;        % length of the sequence
pi = [1/3; 1/3; 1/3]; % inital probability pi_1 = 0.5 and pi_2 =0.5

A  = [0.1 0.3 0.6 ; 0.3 0.1 0.6 ; 0.5 0.2 0.3];

%%alphabet of 6 letters (e.g., a die with 6 sides) E(i,j) is the
E = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t}) 
    1/10 1/10 1/10 1/10 1/10 1/2;
    1/3 1/3 1/12 1/12 1/12 1/12];

[ Y, S ] = HmmGenerateData(N, T, pi, A, E ); 

%%Y is the set of generated observations 
%%S is the set of ground truth sequence of latent vectors 