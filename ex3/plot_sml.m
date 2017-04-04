function [ output_args ] = plot_sml( X, num_labeled)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
label = getlab(X);
X = getdata(X);
labeled = label(1:num_labeled);
X_labeled = X(1:num_labeled,:);
X0 = X_labeled(labeled==0,:);
X1 = X_labeled(labeled==1,:);
X_u = X(num_labeled+1:end,:);
figure;
plot(X0(:,1),X0(:,2),'b*');
hold on
plot(X_u(:,1),X0(:,2),'*');
plot(X1(:,1),X1(:,2),'r+');



end

