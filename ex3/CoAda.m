function [lab_u,index] = CoAda(X,F,num_labeled,T)
%CoAda 此处显示有关此函数的摘要
%   此处显示详细说明
label = getlab(X);
X = getdata(X);
%[N,D]  = size(X);
labeled = label(1:num_labeled);         % get index for labeled data
X_labeled = X(1:num_labeled,:);         % get labeled data
X_unlabeled = X(num_labeled+1:end,:);   % unlabeled data
%X_u = X_unlabeled;
X_u_withi = [X_unlabeled (1:size(X_unlabeled,1))']; 
lab_u = zeros(size(X_u_withi,1),1);
X1 = X_labeled(:,F(1,:));       % view-1
lab1 = labeled;
X2 = X_labeled(:,F(2,:));       % view-2
lab2 = labeled;
[beta1,para1] = adaBoost(X1,T,lab1);
[beta2,para2] = adaBoost(X2,T,lab2);
while size(X_u_withi,1) >=5 
    [beta1,para1] = adaBoost(X1,T,lab1);
    [beta2,para2] = adaBoost(X2,T,lab2);
    X_u = X_u_withi(:,1:end-1);
    [X1_index,X1_lab] = adaTopk(beta1,para1,X_u(:,F(1,:)));
    X2 = [X2;X_u(X1_index,F(2,:))];
    lab2 = [lab2;X1_lab];
    lab_u(X_u_withi(X1_index,end)) = X1_lab;
    X_u_withi(X1_index,:)=[];
    
    X_u = X_u_withi(:,1:end-1);
    [X2_index,X2_lab] = adaTopk(beta2,para2,X_u(:,F(2,:)));
    X1 = [X1;X_u(X2_index,F(1,:))];
    lab1 = [lab1;X2_lab];
    lab_u(X_u_withi(X2_index,end)) = X2_lab;
    X_u_withi(X2_index,:)=[];
end
lab_u1 = adaPredict(beta1,para1,X_unlabeled(:,F(1,:)));
lab_u2 = adaPredict(beta2,para2,X_unlabeled(:,F(2,:)));
%lab_u = lab_u1;
index = (lab_u==lab_u1==lab_u2);
%index = ones(size(X_u_withi,1),1);
end

