% Compare two implementations: One is coordinate descent written by myself,
% another is using Matlab Optimizer to find local minimal.
X0 = [1 2 3 4;4 5 6 7];
X1 = [5 6 7 8 ;8 9 10 11];
lambda = 100;
[m0 m1 cost] = rnmc(X0,X1,lambda,0.0001);
[m0_fmin m1_fmin cost_fmin] = rnmc_fminunc(X0,X1,lambda);

% Test on the exercise dataset
X = load('optdigitsubset.txt');
X0=  X(1:554,:);
X1 = X(555:1125,:);
lambda = 1000000;
[m0 m1 cost] = rnmc(X0,X1,lambda,0.0001);
[m0_fmin m1_fmin cost_fmin] = rnmc_fminunc(X0,X1,lambda);
plot(m0,'g');
hold on;
plot(m0_fmin,'r');