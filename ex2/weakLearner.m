function [ feat,theta,y ] = weakLearner( X,lab)
%weakLearner: A very simple classifier
%  Input: dataset with labels
%  Find the optimal feature f and threshold theta
%  Output: optimal f, theta, and y.

% If a single input(indicating that input is pr_dataset), 
% read in the data and label from pr_dataset.
if nargin < 2
    lab = getlab(X);
    X = getdata(X);
end
[n,f] = size(X);
min_score = 10000000;
for i=1:f
    for j = 1:n
        sign = 0; %sign is: "<"
        % calculate predictions
        Theta = ones(n,1)*X(j);
        predict = X(:,i)-Theta<=0;
        predict1 = X(:,i)-Theta>=0;
        % To match the label: 1 and 2 , every prediction adds one.
        predict1 = predict1+1;
        predict = predict+1;
        score =sum(abs(predict-lab));
        score1 = sum(abs(predict1-lab));
        % Check whether should be ">" or "<"
        if score>score1
            score = score1;
            sign =1;
        end
        % Compare and store the minimum score
        if score<min_score
            min_score = score;
            y = sign;
            feat = i;
            theta = X(j);
        end
    end
end
end


