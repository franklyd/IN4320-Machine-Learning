function [ mean0,mean1,sigma,prior0] = CoLda(X, num_labeled, iterations)
%ldc Linear Discriminate Analysis
%  Useful info
label = getlab(X);
X = getdata(X);
[N,D] = size(X);
labeled = label(1:num_labeled);
X_labeled = X(1:num_labeled,:);
X0 = X_labeled(labeled==0,:);
N0 = size(X0,1);
X1 = X_labeled(labeled==1,:);
N1 = size(X1,1);
X_u = X(num_labeled+1:end,:);

%%% Initialize parameters
prior0 = 0.5;
prior1 = 0.5;
labeled_mean0 = mean(X0,1);
labeled_mean1 = mean(X1,1);
mean0 = labeled_mean0;
mean1 = labeled_mean1; 
% lda: assume the same covarince matrix
labeled_sigma0 = cov(X0);
labeled_sigma1 = cov(X1);
sigma0 = labeled_sigma0;
sigma1 = labeled_sigma1;
sigma = prior0 * sigma0 + prior1 * sigma1 + 1e-5*eye(D);
%p0 = zeros(size(X_u,1),1);
logL = [];

for i = 1: iterations
    %%% E-step: 
    % calculate post_Pr for unlabeled data
    p0 = mvnpdf(X_u, mean0, sigma)+1e-26;
    p1 = mvnpdf(X_u, mean1, sigma)+1e-26;
    %p0
    
    
    %p0_temp = p0;
    %p0(p1==0)= 1e30;    % a vey small positive number
    %p1(p0_temp==0) = 1e30;
    % Log-likelihood p(x,y|theta)
    logL(i) =  sum(log(mvnpdf(X0, mean0, sigma))) ...
        + sum(log(mvnpdf(X1, mean1, sigma)))...
        + sum(log(prior0 .* p0 + prior1 .* p1));
    if i>1
        if abs(logL(i) - logL(i-1)) < 1.0e-4
            break;
        end
    end
    
    % normalize them
    p = prior0 .* p0 + prior1 .* p1;
    p0 = (prior0 .* p0) ./p;
    p1 = 1 - p0;
    p0(p<1e-19) = 0;
    p1(p<1e-19) = 0;
    %p0
    %%% M-step : update para
    % update prior
    %prior0 = (size(X0,1) + sum(p0))/ N;
    %prior1 = (size(X1,1) + sum(p1))/ N;
    % update mean
    %mean0
    %mean1
    mean0 = (labeled_mean0 * N0 + sum((repmat(p0,[1,D]).* X_u),1))/(N0 + sum(p0));
    mean1 = (labeled_mean1 * N1 + sum((repmat(p1,[1,D]).* X_u),1))/(N1 + sum(p1));
    % update sigma
    sigma0 = labeled_sigma0 * N0;
    sigma1 = labeled_sigma1 * N1;
    for j = 1: N - num_labeled
        sigma0 = sigma0 + p0(j).*((X_u(j,:) - mean0)' * (X_u(j,:) - mean0));
        sigma1 = sigma1 + p1(j).*((X_u(j,:) - mean1)' * (X_u(j,:)- mean1));
    end
    sigma0 = sigma0 / (N0 + sum(p0));
    sigma1 = sigma1 / (N1 + sum(p1));
    sigma = prior0 * sigma0 + prior1 * sigma1 + 1.0e-5*eye(D); 

end  
end

