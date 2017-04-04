function [theta,w] = weighted_least_squares(X,yX,Z)
% Compute parameters for a weighted least-squares classifier
%
% Input:
%       X       = source data (n x 1)
%       yX      = source labels (n x 1)
%       Z       = target data (m x 1)
% Output:
%       theta   = classifier parameters (2 x 1)
%       w       = estimated weights (n x 1)
%
% Author: Wouter Kouw
% Last update: 28-03-2017

% Check for nx1 format
if ~iscolumn(X); X = X'; end
if ~iscolumn(Z); Z = Z'; end

% Estimate weights using Kernel Mean Matching
w = kmm(X,Z);
W = diag(w);

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Compute closed-form optimal classifier parameters
theta = (X'*W*X)\X'*W*yX;

end
