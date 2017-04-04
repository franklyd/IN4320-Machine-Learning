%Gaussian D
X = [randn(5000,2);randn(5000,2)+ repmat([1,2],5000,1)];
labels = [zeros(5000,1);ones(5000,1)] + 1;
X_dataset = prdataset(X,labels);
scatterd(X_test);
[X_labeled,X_rest] = gendat(X_dataset,[75,75]);
repeats = 20;

e1_history = zeros(6,repeats);
e1_history1 = zeros(6,repeats);
e1_history2 = zeros(6,repeats);
unlabeled = [1, 20, 160, 320, 640, 1280]; 
T = 5;
%rindex = randperm(57); 
F = [1,2;2,1];

for j=1:repeats
    [X_labeled,X_rest] = gendat(X_dataset,[75,75]);
    [X_test,X_rrest] = gendat(X_rest,[1000,1000]);
    for i = 1:6
        X_unlabeled = gendat(X_rrest,[unlabeled(i),unlabeled(i)]);
        X_train = [X_labeled; X_unlabeled];
        [lab_unlabeled,index] = CoAda(X_train,F,150,T);
        X_u_labeled = prdataset(X_unlabeled(index,:),lab_unlabeled(index,:));% create dataset based on predict label
        X_train1 = [X_labeled; X_u_labeled];
        X_train1 = prdataset(X_train1,getlab(X_train1)-1);
        X_test1 = prdataset(X_test,getlab(X_test)-1);
        %e = X_test1*ldc(X_train1)*testc;
        [mean0, mean1, sigma] = CoLda(X_train1, size(X_train1,1), 300);
        e= lda_predict(X_test1,mean0,mean1,sigma,0.5);
        e1_history(i,j) = e; 
        
        X_train = prdataset(X_train,getlab(X_train)-1);
        % EM algorithm
        [mean0, mean1, sigma,prior0] = lda(X_train, 150, 300);
        e= lda_predict(X_test1,mean0,mean1,sigma,0.5);
        e1_history1(i,j) = e; 
        
        %e = X_test1*ldc(X_train)*testc;
        [mean0, mean1, sigma,prior0] = lda(X_train, 150+2*unlabeled(i), 300);
        [e,ll]= lda_predict(X_test1,mean0,mean1,sigma,prior0);     
        e1_history2(i,j) = e; 
        %LL1(i,j) = ll;
        i
        j
    end
end
figure;
plot(unlabeled,mean(e1_history,2));
hold on
plot(unlabeled,mean(e1_history1,2));
plot(unlabeled,mean(e1_history2,2));
legend('Co-training','EMLDA','oracleLDA');
hold off
