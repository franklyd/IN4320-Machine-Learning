function [ e,predict ] = calculateError( feat,theta,y,X_test,weight,lab_test)
%calculateError calculate the error based on feature,theta and y 
%   
if nargin==4
    lab_test = getlab(X_test);
    X_test = getdata(X_test);
    weight = ones(size(X_test,1),1)/size(X_test,1);
end
if nargin==5
    lab_test = getlab(X_test);
    X_test = getdata(X_test);

end
weight = weight/sum(weight);
n = size(X_test,1);
Theta = ones(n,1)*theta;
if y==0
    predict = X_test(:,feat)-Theta<=0;
    predict = predict+1;
    score = weight'*abs(predict-lab_test);
   
else
    predict = X_test(:,feat)-Theta>=0;
    predict = predict+1;
    score = weight'*abs(predict-lab_test);
    
end
% 
% 
% predict = (Class1(:,feat)- Theta1)>0;
% score = weight1'*predict;
% predict2 = (Class2(:,feat)- Theta2)<=0;
% score2 = weight2'*predict2;
% score = score+score2;
% if y==1
%     score =sum(weight)-score;
% end
e = score;
end

