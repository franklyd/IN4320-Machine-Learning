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
        
end
