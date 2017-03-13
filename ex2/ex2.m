%% a. prove exp(-x) >= (1-x)
%plot it
x = linspace(-5,5,100);
y1 = exp(-x);
y2 = 1-x;
plot(x,y1)
hold on
plot(x,y2);
legend('e^-^x','1-x');
title('e^-^x VS 1-x ')
hold off

%% b&c. weak learner and test the implementation
%randn('seed',0);
A = randn(100,2);
%randn('seed',1);
B = randn(100,2);
B(:,1) = B(:,1)+2;
X = [A;B];
lab = [ones(100,1);ones(100,1)+1];
scatterd(prdataset(X,lab),'legend');
[feat,theta,y] = weakLearner(X,lab);
%%
B(:,1) = B(:,1)*10;
A(:,1) = A(:,1)*10;
X = [A;B];
[feat1,theta1,y1] = weakLearner(X,lab);
figure;
scatterd(prdataset(X,lab),'legend');

%% d testing on digit datset: fisrt 50 as training data
%clear all;
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
%[X,X_test] = gendat(X,[50,50]);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
[feat,theta,y] = weakLearner(X_train);
e1 = calculateError( feat,theta,y,X_test);
%% d test: random split
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
Error = zeros(100,1);
for i=1:100
    [X_train,X_test] = gendat(X,[50,50]);
    [feat,theta,y] = weakLearner(X_train);
    e = calculateError( feat,theta,y,X_test);
    Error(i) = e;
end
 plot(Error,'rx');
title('Scatterplot of the testing errors');
xlabel('number of trials');
ylabel('testing error');

%% e weighted version
% Question: should the weight be normalized??
% a simple case for testing
%X = gendats([5,5],2,2);
%Xb = gendatb([100,100]);
%lab= str2num(getlab(Xb));
%X= pr_dataset(Xb,lab);

scatterd(X,'legend');
weight = [1;1;1;1;1;1*ones(5,1)];
[feat,theta,y]  = weightedWeakLearner( X,weight);
e = calculateError( feat,theta,y,X,weight);

%% e weighted testing on digit dataset
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
lab = getlab(X);
weight = ones(size(lab));
[feat,theta,y]  = weightedWeakLearner( X,weight);

%% implemented adaBoost
X = gendats(100,2,2);
%lab= str2num(getlab(X));
X_data = getdata(X);
X = pr_dataset(X_data,lab);
[X_train,X_test]= gendat(X,0.5);
scatterd(X_train,'legend')
T=100;
%train the classifier
[predLab,beta,para,W] = adaBoost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
% on test set
%predLab_test = adaPredict(beta,para,X_test);
%error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1)
figure;
imagesc(W'),colorbar;
%%
X = gendatb(50,2);
lab= str2num(getlab(X));
X_data = getdata(X);
X_train = pr_dataset(X_data,lab);
figure;
scatterd(X_train,'legend')
T=100;
%train the classifier
[predLab,beta,para,W] = adaBoost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
figure;
imagesc(W'),colorbar;


%% test for adaboost: toy example
A = [1 1;2 1;3 2;4 2;1.5 1];
B = [1 2;2 2;3 1;4 1];
X = pr_dataset([A;B],[1;1;1;1;1;2;2;2;2]);
scatterd(X,'legend')
T=1;
%train the classifier
[predLab,beta,para] = adaBoost(X,T);
feat = para(1)
theta = para(2)
error = sum(abs(predLab-(getlab(X))))/size(X,1);

%% testing for digit datset
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);

%scatterd(X,'legend')
T=5;
%train the classifier
[predLab,beta,para] = adaBoost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
% on test set
predLab_test = adaPredict(beta,para,X_test);
error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1)
%%
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
E_test_history = [];
t_list = [1 5 10 20:2:40 100];
t_list = [1 2 3 5 10 12 14 16 18 20 25 30 35 40];
t_list = [1 10 30 40 50]
for t = t_list
    [predLab,beta,para] = adaBoost(X_train,t);
    error_train = sum(abs(predLab-str2num(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta,para,X_test);
    error_test = sum(abs(predLab_test-str2num(getlab(X_test))))/size(X_test,1);
    E_test_history = [E_test_history error_train];
end
plot(t_list,E_test_history)

%% test
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
T = 200;
E_test_history = zeros(T,1);
E_trn_history = zeros(T,1);
[predLab,beta,para] = adaBoost(X_train,T);
for t = 1:T
    predLab = adaPredict(beta(1:t),para(1:t,:),X_train);
    error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta(1:t),para(1:t,:),X_test);
    error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1);
    E_test_history(t) = error_test;
    E_trn_history(t) = error_train;
end
%%
plot(E_test_history(1:100));
hold on
plot(E_trn_history(1:100));
legend('test','train');
xlabel('T:iterations');
ylabel('error');
title('Testing and training error on different T');

%% Optimal T: 17
[predLab,beta,para,W] = adaBoost(X_train,17);
imagesc(W'),colorbar;

%% show the outlier
X_data = getdata(X_train);
% 66 86
index  =94;
img = reshape(X_data(index,:),[8,8]);
img = mat2gray(img');
figure;
show(img)
