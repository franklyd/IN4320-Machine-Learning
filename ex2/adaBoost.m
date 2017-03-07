function [ predLab,beta,para] = adaBoost( X,T,lab)
%adaBoost train several weak classifiers
% Input: X training data, lab label, T number of iterations
% Output:
if nargin < 3
    lab = getlab(X);
    X = getdata(X);
    %lab(lab==1)=0;
    %lab(lab==2)=1;
end
n = size(X,1);
weight = ones(n,1);
beta = zeros(T,1);
para = zeros(T,3);
for t = 1:T
    p = weight./sum(weight);
    [feat,theta,y] = weightedWeakLearner( X,p,lab);
    para(t,:) = [feat,theta,y];
    [e,pred] = calculateError(feat,theta,y,X,p,lab);
    %question: how to deal with e=0??
    if e==0
        e=0.001;
    end
    beta(t) = e/(1-e);
    weight = weight.*(beta(t).^(1-abs(pred-lab)));
end
predLab = adaPredict(beta,para,X);
end

