function [J, grad] = lrCostFunction(theta, X, y, lambda)
% Bruce Haydon 2022
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);



% The following code is to calculate cost WITHOUT regularization
%J = (1/m)*(-y'*log(h) - (1 - y)'* log(1-h));

% Regularization
% First extract all rows but first in "THETA" and create new vector of (n-1) rows
shift_theta = theta(2:size(theta));
shift_theta

%Now insert a "0" as first row in this new vector to bring size back to "n" rows
%You are now left with copy of original theta vector with "0" replacing first row
%Because first gradient for theta(0) has no regularization term, this matrix will
%be able to be used to multiply directly in calculating regularized gradient.
%
%theta(0) should not be regularized since it is the bias term

reg_theta = [0;shift_theta];
reg_theta

%Regularized cost function

J = (1/m) * (-y'*log(h) - (1 - y)' * log(1-h)) + sum((lambda/(2*m))*reg_theta.^2)

%Regularized gradient function

grad = (1/m) * X' * (h-y)+ (lambda/m)*reg_theta

% =============================================================
% Convert to single column vector
grad = grad(:);

end
