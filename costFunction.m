function [ jVal ] = costFunction( X0,X1,theta,lambda)
%costFunction compute the cost function for a given theta
% This is a function just for the ex1 of ML in TU Delft
dim = size(X0,2);
N0 = size(X0,1);
N1= size(X1,1);
M0 = theta(1:dim);
M1 = theta(dim+1:end);
jVal = sum(sum((X0 - repmat(M0,N0,1)).^2))+ sum(sum((X1 - repmat(M1,N1,1)).^2)) + lambda*sum(abs(M0-M1));
%jVal = sum(sum((X0 - repmat(M0,N0,1)).^2))+ sum(sum((X1 - repmat(M0,N1,1)).^2));
end

