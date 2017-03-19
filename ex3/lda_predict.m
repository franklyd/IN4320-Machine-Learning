function [ error,pred_label] = lda_predict( X,mean0,mean1,sigma)
%lda_predict predict labels and calculate error
%  
label = getlab(X);
X = getdata(X);

p0 = mvnpdf(X, mean0, sigma);
p0(p0==0) = eps(0);
p1 = mvnpdf(X, mean1, sigma);
p1(p1==0) = eps(0);
pred_label = [p1 > p0];
error = sum(abs(pred_label - label))/size(X,1);
end

