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
lambda = 1;
w0 = abs(randn(n,1))+1;
H = 1 / (n^2) * (kernel(X,X) + lambda * eye(n));
f = - 2 / (n * m) * sum(kernel(X,Z),2);
[w,FVAL] = quadprog(H,f,[],[],[],[],zeros(n,1),[],[],options);
disp(FVAL);
end
