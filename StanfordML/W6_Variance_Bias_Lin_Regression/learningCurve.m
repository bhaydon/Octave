function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%
%  LEARNINGCURVE Generates the train and cross validation set errors needed 
%  to plot a learning curve
%  Bruce Haydon (NY, USA)  (https://linkedin.com/in/bhaydon)
%
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   This function computes the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, this should be done in larger intervals.
%
%
% BACKGROUND:
% To plot the learning curve, we need a training and cross validation set
%  error for different training set sizes.
% Specifically, for a training set size of i, you should use the first "i"
% examples (i.e., X(1:i,:) and y(1:i)).
%
% STEP (2) - learn THETA parameters
% "trainlinearreg" function is used to develop the THETA parameters using
% MATLAB/OCTAVE's built-in "fmincg" function
% theta = fmincg(costFunction, initial_theta, options);
%
% After learning the THETA parameters, next step is to compute the error
% on the train-ing and cross validation sets.

% Number of training examples (number rows in matrix)
m = size(X, 1);

% These values will be returned
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ============================================
% Notes  : This function will return training errors in 
%               "error_train" and the cross validation errors in "error_val". 
%               i.e., error_train(i) and 
%               error_val(i) will give errors
%               obtained after training on i examples.
%
% Note: Training error will be evaluated on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, should evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: Using companion function (linearRegCostFunction)
%       to compute the training and cross validation error, function will
%       be called with the lambda argument set to 0. 
%       Note that lambda is still rquired  when running
%       the training to obtain the theta parameters.
%
% Algo: high level description of algorithm used
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% Determine THETA parameters
% lambda is passed on as an input parameter


for i = 1:m;
  
  theta = trainLinearReg(X(1:i), y(1:i), lambda);

  % Note we set lambda=0 for training cost calculation
  [J, grad] = linearRegCostFunction(X(1:i), y(1:i), theta, 0);
  error_train(i) = J;
  
  [J, grad] = linearRegCostFunction(Xval, yval, theta, 0);
   error_val(i) = J;
   
end;
   

end
