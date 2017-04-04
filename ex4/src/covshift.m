function covshift()
% Visualize a least-squares and a weighted least-squares classifier
% in a covariate shift setting
%
% Input:
%       -
% Output:
%       -
% Author: Wouter Kouw
% Last update: 28-03-2017

% Target variance
s2 = 2;

% Classes and priors
Y = [-1 1];
pyn = 1./2;
pyp = 1./2;

% Class-posteriors p_S(y|x) (cumulative normal distribution)
pyn_X = @(a) (1+erf(-a./sqrt(2)))./2;
pyp_X = @(a) (1+erf( a./sqrt(2)))./2;

% Source data marginal p_S(x) (normal distribution)
pX = @(a) normpdf(a, 0, 1);

% Source class-conditional likelihoods p_S(x|y) 
pX_yn = @(a) pyn_X(a) .* pX(a)./pyn;
pX_yp = @(a) pyp_X(a) .* pX(a)./pyp;

% Class-posteriors p_T(y|x) (cumulative normal distribution)
pyn_Z = @(b) (1+erf(-b./sqrt(2)))./2;
pyp_Z = @(b) (1+erf( b./sqrt(2)))./2;

% Target data marginal p_T(x) (normal distribution)
pZ = @(b) normpdf(b, 0, sqrt(s2));

% Target class-conditional likelihoods p_T(x|y)
pZ_yn = @(b) pyn_Z(b) .* pZ(b)./pyn;
pZ_yp = @(b) pyp_Z(b) .* pZ(b)./pyp;

%% Rejection sampling of data

% Amount of samples from each domain
n = 1e2;
m = 1e2;

% Sampling range limits
xl = [-50 50];
zl = [-50 50];

% Rejection sampling of source data
Xy_n = sampleDist(pX_yn,1./sqrt(2*pi),round(n.*pyn),[xl(1) xl(2)], false);
Xy_p = sampleDist(pX_yp,1./sqrt(2*pi),round(n.*pyp),[xl(1) xl(2)], false);

% Rejection sampling of target data
Zy_n = sampleDist(pZ_yn,1./sqrt(2*pi*s2),round(m.*pyn),[zl(1) zl(2)], false);
Zy_p = sampleDist(pZ_yp,1./sqrt(2*pi*s2),round(m.*pyp),[zl(1) zl(2)], false);

% Concatenate to datasets
X = [Xy_n; Xy_p];
Z = [Zy_n; Zy_p];
yX = [-ones(size(Xy_n,1),1); ones(size(Xy_p,1),1)];
yZ = [-ones(size(Zy_n,1),1); ones(size(Zy_p,1),1)];

%% Compute classifiers

% Compute the source least-squares classifier
theta = least_squares(X,yX);

% Compute the source weighted least-squares classifier
[theta_weighted,w] = weighted_least_squares(X,yX,Z);

%% Visualization parameters

% Range limits
rl = [-3,+3];

% Steps of x
uu = linspace(rl(1),rl(2),101);

% Font size
fS = 20;

% Marker size
mS = 10;

% Line width
lW = 4;

%% Visualize classifiers on the source and target domains

figure()
subplot(1,2,1);

% Plot source class-conditional likelihoods
plot(uu, pX_yn(uu)+Y(1), 'r', 'LineWidth', lW);
hold on
plot(uu, pX_yp(uu)+Y(2), 'b', 'LineWidth', lW);

% Plot samples from source distribution
plot(X(yX==Y(1)),Y(1), 'rx', 'MarkerSize', mS);
hold on
plot(X(yX==Y(2)),Y(2), 'bo', 'MarkerSize', mS);

% Plot classifiers
fplot(@(x) x*theta(1)+theta(2), 'Color', 'k', 'LineWidth', lW);
fplot(@(x) x*theta_weighted(1)+theta_weighted(2), '--', 'Color', 'c', 'LineWidth', lW);

% Visualization settings
title(['Source domain']);
lines = findobj(gcf, 'Type', 'FunctionLine');
legend(lines, {'wls','ls'}, 'Location', 'northwest');

xlabel('x');
ylabel('y');

set(gca, 'YLim', [-1.5 2.5], 'XLim', [rl(1) rl(2)], 'FontSize', fS, 'FontWeight', 'bold');
axis square

subplot(1,2,2);

% Plot target class-conditional likelihoods
plot(uu, pZ_yn(uu)+Y(1), 'r', 'LineWidth', lW);
hold on
plot(uu, pZ_yp(uu)+Y(2), 'b', 'LineWidth', lW);

% Plot samples from target distribution
plot(Z(yZ==Y(1)),Y(1), 'rx', 'MarkerSize', mS);
hold on
plot(Z(yZ==Y(2)),Y(2), 'bo', 'MarkerSize', mS);

% Plot classifiers
fplot(@(x) x*theta(1)+theta(2), 'Color', 'k', 'LineWidth', lW);
fplot(@(x) x*theta_weighted(1)+theta_weighted(2), '--', 'Color', 'c', 'LineWidth', lW);

% Visualization parameters
title(['Target domain']);
lines = findobj(gcf, 'Type', 'FunctionLine');
legend(lines, {'wls','ls'}, 'Location', 'northwest');

xlabel('x');
ylabel('y');

set(gca, 'YLim', [-1.5 2.5], 'XLim', [rl(1) rl(2)], 'FontSize', fS, 'FontWeight', 'bold');
axis square

set(gcf, 'Color', 'w', 'Position', [100 100 2000 800]);

saveas(gcf, 'scatter.png'); 

%% Visualize weight histogram

figure()

% Visualize histogram
histogram(w)

% Visualization options
title(['Weight distribution']);

xlabel('w');
ylabel('Frequency');

set(gca, 'FontSize', fS, 'FontWeight', 'bold', 'YLim', [0 n]);
set(gcf, 'Color', 'w', 'Position', [100 100 1000 800])

saveas(gcf, 'histogram_weights.png');

end

