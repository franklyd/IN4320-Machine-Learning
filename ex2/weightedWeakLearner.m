function [ feat,theta,y ] = weightedWeakLearner( X,weight,lab)
%weakLearner: A very simple classifier
%  Input: dataset with labels
%  Find the optimal feature f and threshold theta
%  Output: optimal f, theta, and y.
if nargin < 3
    lab = getlab(X);
    X = getdata(X);
    
end
[n,f] = size(X);
Class1 = X(lab==1,:);
weight1 = weight(lab==1,:);
n1 = size(Class1,1);
Class2 = X(lab==2,:);
weight2 = weight(lab==2,:);
n2 = size(Class2,1);
max_score = 0;
for i=1:f
    for j = 1:n
        sign = 0; %sign is: >
        Theta1 = zeros(n1,1) +X(j);
        Theta2 = zeros(n2,1) +X(j);
        predict1 = (Class1(:,i)- Theta1)>0;
        score1 = weight1'*predict1;
        predict2 = (Class2(:,i)- Theta2)<=0;
        score2 = weight2'*predict2;
        score = score1 + score2;
        if sum(weight)-score > score
            score = sum(weight)-score;
            sign=1; %sign set to: <
        end
        if score>max_score
            max_score = score;
            y = sign;
            feat = i;
            theta = X(j);
        end
    end
end
end


