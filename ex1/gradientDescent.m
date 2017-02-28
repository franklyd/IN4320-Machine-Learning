function [M0,M1, J_history] = gradientDescent(X0, X1,alpha,lambda,num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
dim = size(X0,2);
N0 = size(X0,1);
N1= size(X1,1);
M0 = rand(1,dim)+15;
M1 = rand(1,dim)-1;
J_history = zeros(num_iters, 1);
count = 1;
for iter = 1:num_iters
    M0_temp = M0;
    M0 = M0 - alpha*(2*N0*M0 - 2*sum(X0)+ lambda*(2./(exp(-(M0-M1))+1) -1));
    M1 = M1 - alpha*(2*N1*M1 - 2*sum(X1)+ lambda*(2./(exp(-(M1-M0_temp))+1) -1));
    % Save the cost J in every iteration    
    J_history(iter) = costFunction(X0,X1,[M0 M1],lambda);
    plot(M0(1),M1(1),'rx', 'MarkerSize', count+1, 'LineWidth', 2);
end

end
