%% 1(a) Draw the lost function
J_vals = [];
m1_vals = linspace(-2,2,100);
lambda = [0,2,4,6];
for i = 1:4 
    for j = 1:length(m1_vals)
        J_vals(i,j) = (-1-m1_vals(j))^2 +(1 - m1_vals(j))^2 + lambda(i)*norm((1-m1_vals(j)));
    end
    subplot(2,2,i);
    plot(m1_vals,J_vals(i,:));
    xlabel('m+');
    ylabel('loss function');
    leg = strcat('lambda = ',num2str(lambda(i)));
    legend(leg);
end

%% 2b Draw the contour
m1_vals = linspace(-10,10,100);
m0_vals = linspace(-10,10,100);
X0 = [-1;-2;-3;-4;-5];
X1 = [1;2;3;4;5];
J_vals = zeros(length(m0_vals), length(m1_vals));
lambda = 500;
% Fill out J_vals
for i = 1:length(m0_vals)
    for j = 1:length(m1_vals)
	  t = [m0_vals(i); m1_vals(j)];    
	  J_vals(i,j) = costFunction( X0,X1,t,lambda);
    end
end
J_vals = J_vals';
% Surface plot
figure;
mesh(m0_vals, m1_vals, J_vals);
xlabel('m_-'); ylabel('m_+');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(m0_vals, m1_vals, J_vals, linspace(0.0001, 300, 30))
xlabel('m_-'); ylabel('m_+');
hold on;
[m0,m1] = rnmc(X0,X1,lambda,0.0001);
plot(m0, m1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%% 3a
