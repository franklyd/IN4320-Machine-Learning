function [ error,logL] = lda_predict( X,mean0,mean1,sigma,prior0)
%lda_predict predict labels and calculate error
%
label = getlab(X);
X = getdata(X);

p0 = mvnpdf(X, mean0, sigma)+eps(0);
p1 = mvnpdf(X, mean1, sigma)+eps(0);

% p0_temp = p0;
% p0(p1==0)= eps;    % a vey small positive number
% p1(p0_temp==0) = eps;

pred_label = [(1-prior0)*p1 > prior0*p0];
error = sum(abs(pred_label - label))/size(X,1);
%logL = sum(log(prior0 .* p0 + (1 - prior0) .* p1));
logL = -sum(log(mvnpdf(X(label==0,:), mean0, sigma)+eps(0)))-sum(log(mvnpdf(X(label==1,:), mean1, sigma)+eps(0)));
%logL
end

