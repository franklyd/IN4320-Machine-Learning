%% read in data
Data = load('spambase.data');
X = Data(:,1:57);
X_norm = X./repmat(std(X,1),[4601,1]);
% todo: normalize it
label = Data(:,58);
X_dataset = prdataset(X_norm,label);

%% basic LDA testing
[X_labeled,X_test] = gendat(X_dataset,[75,75]);
X_unlabeled = gendat(X_test,[64,64]);
X = [X_labeled; X_unlabeled];
[mean0, mean1, sigma] = lda(X, 150, 25);
[e, lab] = lda_predict(X_test,mean0,mean1,sigma);
%% Visualize Covariance Matrix
imagesc(cov(X0));
colorbar;

%%
e_history = zeros(8,10);
unlabeled = [0, 10, 20, 40, 80, 160, 320, 640, 1280]; 
[X_labeled,X_test] = gendat(X_dataset,[75,75]);
for i=1:9
    for j = 1:10
        X_unlabeled = gendat(X_test,[unlabeled(i),unlabeled(i)]);
        X = [X_labeled; X_unlabeled];
        [mean0, mean1, sigma] = lda(X, 150, 200);
        e = lda_predict(X_test,mean0,mean1,sigma);
        e_history(i,j) = e; 
    end
end
plot(unlabeled,e_history,'*')
    
    
