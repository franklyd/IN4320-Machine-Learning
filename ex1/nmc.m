X = load('optdigitsubset.txt');
class0 = ones(554,1);
class1 = ones(571,1).*(-1);
X0=  X(1:554,:);
X1 = X(555:1125,:);

X0 = [1 2 3];
X1 = [7 8 9];
lambda = 200;
tolerence = 0.001;
size0 = size(X0);
size1 = size(X1);
D = size0(2);
N0 = size0(1);
N1= size1(1);
M0 = [4 5 6];
M1 = [4 5 6];
[m0 m1 l]=rnmc( X0,X1,lambda,tolerence);
lost = sum(sum((X0 - repmat(M0,N0,1)).^2))+ sum(sum((X1 - repmat(M1,N1,1)).^2)) + lambda*sum(abs(M0-M1))

