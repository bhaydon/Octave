function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%
%% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%  regression with multiple variables
%  Bruce Haydon (https://linkedin/in/bhaydon)
%
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialization
% m = length(y); % number of training examples
[m n] = size(X); % m= number of training examples in set, n= #features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ===========================================
% Notes: Computes the cost and gradient of regularized linear 
%        regression for a particular choice of theta.
%
%        J is set to the cost and grad to the gradient.
%
% Have to insert a column of ones to the left to mult by theta(0)
% X_=[ones(m,1) X]  This is already looked after by calling function

% Calculate predicted value
h = X * theta;  

J_unreg = sum( ((h-y) .^ 2 ) /(2*m));

% Calculate regularization parameter

Reg = (lambda/(2.*m)) * sum(theta(2:n).^2);

% Add regularization parameter to existing cose
J = J_unreg + Reg;

% compute gradient under two cases: theta(0) which has no regularization, 
% and all other thetas which do.

% Note that we should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta, this corresponds to the first column.
% Therefore, remove first columm on each matrix

% Compute gradient without regularization
grad_unreg = ( X' * (h-y)) / m ;

grad = grad_unreg;

% Apply regularisation to the gradient, omitting first partial derivative
grad(2:n) += lambda * theta(2:n) / m

% =========================================================================

grad = grad(:);

end
