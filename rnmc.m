function [ M0,M1,new_lost ] = rnmc( X0,X1,lambda,tolerence)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
dim = size(X0,2);
sizeX0 = size(X0,1);
sizeX1= size(X1,1);
M0 = mean(X0);
%M0 = rand(1,D);
M1 = mean(X1);
%M1 = rand(1,D);
new_lost = sum(sum((X0 - repmat(M0,sizeX0,1)).^2))+ sum(sum((X1 - repmat(M1,sizeX1,1)).^2)) + lambda*sum(abs(M0-M1));
max_step = new_lost;
lost_vector = [new_lost];
if sizeX0 == 1
    M0 = X0;
end
if sizeX1 == 1
    M1 = X1;
end

while max_step >= tolerence
    new_lost = sum(sum((X0 - repmat(M0,sizeX0,1)).^2))+ sum(sum((X1 - repmat(M1,sizeX1,1)).^2)) + lambda*sum(abs(M0-M1));
    max_step = 0;
    for i = 1:2*dim
        lost = new_lost;
        %compute for m0
        if i <= dim
             M = mean(X0(:,i));
            if M < M1(i) - lambda/(2*sizeX0)
                M0(i) = lambda/(2*sizeX0) + M;
            elseif M > M1(i) + lambda/(2*sizeX0)
                M0(i) = -lambda/(2*sizeX0) + M;
            else
                %M0(i) = (M1(i)+M0(i))/2;
                %M0(i) = M1(i);
                M0(i) = (M*sizeX0+sum(X1(:,i)))/(sizeX0+sizeX1);
                %M0(i) = (M+mean(X1(:,i)))/2;
            end
        else
            M = mean(X1(:,i-dim));
            if M < M0(i-dim) - lambda/(2*sizeX1)
                M1(i-dim) = lambda/(2*sizeX1) + M;
            elseif M > M0(i-dim) + lambda/(2*sizeX1)
                M1(i-dim) = -lambda/(2*sizeX1) + M;
            else
                %M1(i-D) = (mean(X0(:,i-D))+ M)/2;
                M1(i-dim) = (M*sizeX1+sum(X0(:,i-dim)))/(sizeX1+sizeX0);
                %M1(i-D) = M0(i-D);
            end
            new_lost = sum(sum((X0 - repmat(M0,sizeX0,1)).^2))+ sum(sum((X1 - repmat(M1,sizeX1,1)).^2)) + lambda*sum(abs(M0-M1));
            %lost_vector = [lost_vector new_lost];
            if lost - new_lost > max_step
                max_step = lost - new_lost;
            end
        end
    end
end
end             

