function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
X
theta
h = sigmoid(X*theta);
h


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
grad
grad = grad(:);
grad
end
