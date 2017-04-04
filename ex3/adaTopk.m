function [ X_index,X_lab ] = adaTopk(beta,para,X)
%adaPredict: Predict labels for adaboosting classifiers
% Input: beta(T,1), para(T,3), X(n,f)  
% para: [feat,theta,y]
% Output: predLab(n,1)

%Get useful info
N = size(X,1);
T = size(beta,1);
if N>=10
    k = 5;
elseif N==1
    k = 0;
else
    k = 1;
end

%X_index = zeros(3,1);
%X_lab = zeros(3,1);

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
% Predict labels (0 for class1, 1 for class2)
total_scores = sum(scores,2);
predLab = total_scores>=baseScore;
predLab = predLab+1;
total_scores(total_scores>baseScore) = 2*baseScore - total_scores(total_scores>baseScore);
[~,index] = sort(total_scores);
%X_index = [index(1:k);index(end-k+1:end)];
X_index = index(1:k);
X_lab = predLab(X_index);

%X_lab
% topk = total_scores(index(1:k));
% 
% for i = 1:3
%     if topk(i)<=3
%         X_index(i) = index(topk(i));
%         X_lab(i) = 0;
%     else
%         X_index(i) = index(end - 6 + top(i));
%         X_lab(i) = 1;
%     end
% end
% Tranform labels to 1 for class1 ,2 for class2 
%predLab = predLab+1;
end

