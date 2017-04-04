function [theta] = least_squares(X,yX)
% Compute parameters for a least-squares classifier
%
% Input:
%       X       = source data (n x 1)
%       yX      = source labels (n x 1)
% Output:
%       theta   = classifier parameters (2 x 1)
%
% Author: Wouter Kouw
% Last update: 28-03-2017

% Check for nx1 format
if ~iscolumn(X); X = X'; end

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Compute closed-form optimal classifier parameters
theta = (X'*X)\X'*yX;

end
