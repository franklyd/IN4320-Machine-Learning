function [ predLab,beta,para,W] = adaBoost( X,T,lab)
%adaBoost train several weak classifiers
% Input: X training data, lab label, T number of iterations
% Output: predicted label, beta and parameters for all base classifiers
if nargin < 3
    lab = getlab(X);
    X = getdata(X);
end
%useful info
n = size(X,1);
% Initialize weight, beta and parameters matrix
weight = ones(n,1);
W = ones(n,T);
beta = zeros(T,1);
para = zeros(T,3);
% run T times to train T classifiers
for t = 1:T
    %W(:,t) = weight;
    p = weight./sum(weight);
    W(:,t) = p;
    [feat,theta,y] = weightedWeakLearner( X,p,lab);
    para(t,:) = [feat,theta,y];
    [e,pred] = calculateError(feat,theta,y,X,p,lab);
    % if e==0, set e to a really small number to avoid error
    if e==0
        e=0.00001;
    end
    beta(t) = e/(1-e);
    weight = weight.*(beta(t).^(1-abs(pred-lab)));
end
%After T iterations, predict labels using T weighted classifiers
predLab = adaPredict(beta,para,X);
end

