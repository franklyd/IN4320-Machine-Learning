%% read in data
Data = load('spambase.data');
X = Data(:,1:57);
X_norm = X./repmat(std(X,1),[4601,1]);   % normalization process
label = Data(:,58);
X_dataset = prdataset(X_norm,label);

%% basic LDA testing
%[X_labeled,X_rest] = gendat(X_dataset,[75,75]);
numU =0;
%[X_test,X_rrest] = gendat(X_rest,[450,450]);
X_unlabeled = gendat(X_rrest,[numU,numU]);
X = [X_labeled; X_unlabeled];
[mean0, mean1, sigma,prior0] = lda(X,150, 500);
[e, ll] = lda_predict(X_test,mean0,mean1,sigma,prior0);

%% Visualize Covariance Matrix
imagesc(cov(X_norm));
colorbar;

%%
repeats = 10;
e_history = zeros(9,repeats);
LL = zeros(9,repeats);
e_history1 = zeros(9,repeats);
LL1 = zeros(9,repeats);
unlabeled = [0, 10, 20, 40, 80, 160, 320, 640, 1280]; 
label = Data(:,58)+1;
X_dataset = prdataset(X_norm,label);
%[X_labeled,X_test] = gendat(X_dataset,[75,75]);
%[X_labeled,X_rest] = gendat(X_dataset,[75,75]);
%[X_test,X_rrest] = gendat(X_rest,[450,450]);
%X_unlabeled = gendat(X_rrest,[numU,numU]);

% X = [randn(5000,2);randn(5000,2)+ repmat([1.5,-1.3],5000,1)];
% labels = [zeros(5000,1);ones(5000,1)];
% X_dataset = prdataset(X,labels);
for j=1:repeats
    [X_labeled,X_rest] = gendat(X_dataset,[75,75]);
    [X_test,X_rrest] = gendat(X_rest,[450,450]);
    for i = 1:9
        X_unlabeled = gendat(X_rrest,[unlabeled(i),unlabeled(i)]);
        X = [X_labeled; X_unlabeled];
        [mean0, mean1, sigma,prior0] = lda(X, 150, 300);
        [e,ll]= lda_predict(X_test,mean0,mean1,sigma,prior0);
        e_history(i,j) = e; 
        LL(i,j) = ll;
        [mean0, mean1, sigma,prior0] = lda(X, 150+2*unlabeled(i), 300);
        [e,ll]= lda_predict(X_test,mean0,mean1,sigma,prior0);
        e_history1(i,j) = e; 
        LL1(i,j) = ll;
    end
end
plot(unlabeled,mean(e_history,2));
%errorbar(unlabeled,mean(e_history,2),std(e_history')/2)
hold on
plot(unlabeled,mean(e_history1,2));
%errorbar(unlabeled,mean(e_history1,2),std(e_history1')/2);
legend('EMLDA','oracleLDA');
hold off
figure;
plot(unlabeled,mean(LL,2));
%hold on
%plot(unlabeled,mean(LL1,2),'*');
%legend('EMLDA','oracleLDA');
%hold off

%% Co-training
Data = load('spambase.data');
X = Data(:,1:57);
X_norm = X./repmat(std(X,1),[4601,1]);   % normalization process
label = Data(:,58)+1;
X_dataset = prdataset(X_norm,label);
repeats = 25;
%e_history = zeros(9,repeats);
%LL = zeros(9,repeats);
%e_history1 = zeros(9,repeats);
%LL1 = zeros(9,repeats);
%e_history2 = zeros(9,repeats);
unlabeled = [1, 10, 20, 40, 80, 160, 320, 640, 1280]; 
T = 50;

rindex = randperm(57); 
F = [rindex(1:29);rindex(29:end)];
F = F1;

for j=16:repeats
    [X_labeled,X_rest] = gendat(X_dataset,[75,75]);
    [X_test,X_rrest] = gendat(X_rest,[450,450]);
    for i = 1:9
        X_unlabeled = gendat(X_rrest,[unlabeled(i),unlabeled(i)]);
        X_train = [X_labeled; X_unlabeled];
        [lab_unlabeled,index] = CoAda(X_train,F,150,T);
%         if i>=2
%         sum(abs(lab_unlabeled(index,:) - getlab(X_unlabeled(index,:))))/(sum(index))
%         end
        X_u_labeled = prdataset(X_unlabeled(index,:),lab_unlabeled(index,:));% create dataset based on predict label
        X_train1 = [X_labeled; X_u_labeled];
        X_train1 = prdataset(X_train1,getlab(X_train1)-1);
        X_test1 = prdataset(X_test,getlab(X_test)-1);
        [mean0, mean1, sigma] = CoLda(X_train1, size(X_train1,1), 100);
        [e,ll]= lda_predict(X_test1,mean0,mean1,sigma,0.5);
        e_history(i,j) = e; 
        [i,j,e]
        LL(i,j) = ll;
        
        X_train = prdataset(X_train,getlab(X_train)-1);
        % EM algorithm
        [mean0, mean1, sigma,prior0] = lda(X_train, 150, 300);
        [e,ll]= lda_predict(X_test1,mean0,mean1,sigma,prior0);
        e_history1(i,j) = e; 
        LL1(i,j) = ll;
        
        %e = X_test1*ldc(X_train)*testc;
        [mean0, mean1, sigma,prior0] = lda(X_train, 150+2*unlabeled(i), 300);
        [e,ll]= lda_predict(X_test1,mean0,mean1,sigma,prior0);     
        e_history2(i,j) = e; 
        %LL1(i,j) = ll;
        %i
        %j
    end
end
figure;
plot(unlabeled,mean(e_history,2));
hold on
plot(unlabeled,mean(e_history1,2));
plot(unlabeled,mean(e_history2,2));
legend('Co-training','EMLDA','oracleLDA');
hold off
figure;
plot(unlabeled,mean(LL,2));
hold on
plot(unlabeled,mean(LL1,2));
legend('Co-training','EMLDA');
hold off

%% Testing on a Gassian Distribution
%X = gendats([5000,5000],40,4);
%X = [randn(5000,10);randn(5000,10)+ repmat([0.5,0.3,0.3,-0.5,0.4,-0.5,-0.5,-0.9,-1.5,-1],5000,1)];
X = [randn(5000,2);randn(5000,2)+ repmat([1.5,-1],5000,1)];
labels = [zeros(5000,1);ones(5000,1)];
X_dataset = prdataset(X,labels);
scatterd(X_dataset);
[X_labeled,X_rest] = gendat(X_dataset,[75,75]);
numU =400;
[X_test,X_rrest] = gendat(X_rest,[450,450]);
X_unlabeled = gendat(X_rrest,[numU,numU]);
X_train = [X_labeled; X_unlabeled];
[mean0, mean1, sigma,prior0] = lda(X_train, 150, 500);
[e, ll] = lda_predict(X_test,mean0,mean1,sigma,prior0)

%% testing on a no-gassian D
T = 5;
%rindex = randperm(57); 
F = [rindex(1:29);rindex(29:end)];
lab_unlabeled = CoAda(X,F,150);

%% adaboost co-training
Data = load('spambase.data');
X = Data(:,1:57);
X_norm = X./repmat(std(X,1),[4601,1]);   % normalization process
label = Data(:,58)+1;
X_dataset = prdataset(X_norm,label);
[X_labeled,X_test] = gendat(X_dataset,[75,75]);
numU =100;
X_unlabeled = gendat(X_test,[numU,numU]);
X = [X_labeled; X_unlabeled];

T = 5;
%rindex = randperm(57); 
F = [rindex(1:29);rindex(29:end)];
lab_unlabeled = CoAda(X,F,150,T);
%sum(abs(lab_unlabeled - getlab(X_unlabeled)))/40;

 X_u_labeled = prdataset(X_unlabeled,lab_unlabeled);
 X = [X_labeled; X_u_labeled];
 X = prdataset(X,getlab(X)-1);
 [mean0, mean1, sigma,prior0] = lda(X, 150+numU*2, 500);
X_test = prdataset(X_test,getlab(X_test)-1);
e= lda_predict(X_test,mean0,mean1,sigma,prior0)

%%
%X = gendatd([5000,5000],,2);
X = [randn(5000,2);randn(5000,2)+ repmat([1.5,2.5],5000,1)];
labels = X(:,2)>0;
%labels = X(:,2)> X(:,1);
labels = labels + 1;
%labels = [zeros(5000,1);ones(5000,1)] +1;
X_dataset = prdataset(X,labels);
%scatterd(X_dataset);
[X_labeled,X_rest] = gendat(X_dataset,[75,75]);
[X_test,X_rrest] = gendat(X_rest,[1000,1000]);
numU =1;
X_unlabeled = gendat(X_rrest,[numU,numU]);
X_train = [X_labeled; X_unlabeled];
T = 5;
rindex = randperm(2); 
%F = [rindex(1);rindex(2)];
F = [1,2;2,1];
[lab_unlabeled,index] = CoAda(X_train,F,150,T);
sum(abs(lab_unlabeled(index,:) - getlab(X_unlabeled(index,:))))/(sum(index))
X_u_labeled = prdataset(X_unlabeled(index,:),lab_unlabeled(index,:));% create dataset based on predict label
X_train1 = [X_labeled; X_u_labeled];
X_train1 = prdataset(X_train1,getlab(X_train1)-1);
[mean0, mean1, sigma,prior0] = CoLda(X_train1, size(X_train1,1), 100);
X_test1 = prdataset(X_test,getlab(X_test)-1);
e= lda_predict(X_test1,mean0,mean1,sigma,0.5);
%e = X_test1*ldc(X_train1)*testc
e_lower = X_test*ldc(X_train)*testc
e_lowest = X_test*ldc(X_test)*testc
