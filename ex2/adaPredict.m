function [ predLab ] = adaPredict(beta,para,X)
%adaPredict: Predict labels for adaboosting classifiers
% Input: beta [T,1], para[T,3] X[n,f]  
% para: [feat,theta,y]
% Output: predLab[n,1]
N = size(X,1);
T = size(beta,1);
baseScore = 0.5*sum(log(1./beta));
scores = zeros(N,T);
for t=1:T
    feat = para(t,1);
    theta = para(t,2);
    y = para(t,3);
    Theta = ones(N,1)*theta;
    if y==0
        scores(:,t) =X(:,feat)-Theta<=0;
    else
        scores(:,t) =  X(:,feat)-Theta>=0;
    end
    scores(:,t) = scores(:,t)*log(1/beta(t));
end
predLab = sum(scores,2)>=baseScore;
predLab = predLab+1;
end

