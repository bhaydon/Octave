function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Computes the mean of the data and the variances
%                "mu(i)"s contains the mean of
%               the data for the i-th feature and sigma2(i)
%               contains variance of the i-th feature.
%

for i = 1:n
  mu(i) = (1./m)*sum(X(:,i));
endfor

for i = 1:n
  sigma2(i) = (1./m) * sum( (X(:,i)-mu(i)).^2);
endfor





% =============================================================


end
