function [ e ] = calculateError( feat,theta,y,X_test,lab_test)
%calculateError calculate the error based on feature,theta and y 
%   
if nargin < 5
    lab_test = getlab(X_test);
    X_test = getdata(X_test);
end
[n,f] = size(X_test);
Class1 = X_test(lab_test==1,:);
n1 = size(Class1,1);
Class2 = X_test(lab_test==2,:);
n2 = size(Class2,1);

Theta1 = ones(n1,1)*theta;
Theta2 = ones(n2,1)*theta;

score1 = sum((Class1(:,feat)- Theta1)>0);
score2 = sum((Class2(:,feat)- Theta2)<=0);
score = score1+score2;
if y==1
    score = n - score;
end
e = score/n;
end

