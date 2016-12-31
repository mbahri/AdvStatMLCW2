% Example with three gaussians

N  = 1000;         % number of sequences
T  = 100;        % length of the sequence
pi = [1/3; 1/3; 1/3]; % inital probability pi_1 = 0.5 and pi_2 =0.5

A  = [0.1 0.3 0.6 ; 0.3 0.1 0.6 ; 0.5 0.2 0.3];         %p(y_t|y_{t-1})


%%one dimensional Gaussians 

E.mu    =[ .1 5 20]; %%the means of each of the Gaussians
E.sigma2=[ .4 .6 .8]; %%the variances
   

 
[ Y, S ] = HmmGenerateData(N, T, pi, A, E, 'normal'); 

%%Y is the set of generated observations 
%%S is the set of ground truth sequence of latent vectors 
