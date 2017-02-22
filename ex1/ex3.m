X = load('optdigitsubset.txt');
X0=  X(1:554,:);
X1 = X(555:1125,:);
lambda = 10;
[m0 m1 cost] = rnmc(X0,X1,lambda,0.0001);
img0 = reshape(m0,[8,8]); %resize the images
img0 = mat2gray(img0);

img1 = reshape(m1,[8,8]); %resize the images
img1 = mat2gray(img1);
show(img0);
%show(img1);
