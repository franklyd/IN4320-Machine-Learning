function [w] = kmm(X,Z)
% Estimate weights using Kernel Mean Matching
% Paper: Huang, Smola, Gretton, Borgwardt, Schoelkopf (2006)
% Correcting Sample Selection Bias by Unlabeled Data
%
% Input:
%       X       = source data (n x 1)
%       Z       = target data (m x 1)
% Output:
%       w       = weights (n x 1)
%
% Author: Wouter Kouw
% Last update: 28-03-2017

% Sizes
n = size(X,1);
m = size(Z,1);

% Optimization options
options = optimoptions('quadprog', 'Display', 'final', ...
    'StepTolerance', 1e-5, ...
    'maxIterations', 1e2);

%%%%%%%%%%%%%%%%%%%%%%%
%%%% Add your code here

% set hyperparameters
lambda = 0;
epsilon = 1e-2;

% set parameters for quadprog solver
H = 1 / (n ^ 2) * (kernel(X,X) + lambda * eye(n));
f = - 2 / (n * m) * sum(kernel(X,Z),2);
 
% apply the constraint
A = 1/n * [ones(1,n); - ones(1,n)];
b = [epsilon + 1; epsilon - 1];

% quadprog solver
[w,FVAL] = quadprog(H,f,A,b,[],[],zeros(n,1),[],[],options);

% for debugging
disp(FVAL);
end
