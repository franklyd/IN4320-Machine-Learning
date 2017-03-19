%% read in data
Data = load('spambase.data');
X = Data(:,1:57);
X_norm = X./repmat(std(X,1),[4601,1]);
% todo: normalize it
label = Data(:,58);
X_dataset = prdataset(X_norm,label);

%% basic LDA testing
%[X_labeled,X_test] = gendat(X_dataset,[75,75]);
%X_unlabeled = gendat(X_test,[10,10]);
%X = [X_labeled; X_unlabeled];
[m0, m1, sigma] = lda(X, 150, 35);
%% Visualize Covariance Matrix
imagesc(cov(X0));
colorbar;