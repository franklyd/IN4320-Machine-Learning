function [ output_args ] = adaBoost( X,y,T)
%adaBoost train several weak classifiers
% Input: X training data, y label, T number of iterations
% Output:
if nargin < 3
    lab = getlab(X);
    X = getdata(X);
    %lab(lab==1)=0;
    %lab(lab==2)=1;
end
[n,f] = size(X);
weight = ones(n,1);
for t = 1:T
    p = weight/sum(weight);
    [feat,theta,y] = weightedWeakLearner( X,p,lab);
    if y==0
        predY = 
    end
    
end
    


end

