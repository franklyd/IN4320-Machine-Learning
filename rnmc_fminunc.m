function [M0,M1,cost,exitflag ] = rnmc_fminunc( X0,X1,lambda)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
dim = size(X0,2);
initialTheta = [mean(X0) mean(X1)];
[theta, cost,exitflag] = fminunc(@(t)(costFunction(X0,X1,t,lambda)),initialTheta);
M0 = theta(1:dim);
M1 = theta(dim+1:end);
end

