function [ out ] = kernel( X1,X2 )
% kernel: apply the kernel to X1, X2
%
n = size(X1,1);
m = size(X2,1);
out = zeros(n,m);
for i = 1:n
    out(i,:) = exp(-(X1(i) - X2).^2 / 2);
end

end

