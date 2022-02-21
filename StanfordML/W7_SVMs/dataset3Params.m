function [C, sigma] = dataset3Params(X, y, Xval, yval)
%
%DATASET3PARAMS returns tbe best choice of C and sigma
%where the optimal (C, sigma) learning parameters are used for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns  C and 
%   sigma. This function attempts to return the optimal C and 
%   sigma based on a cross-validation set.
%   Bruce Haydon New York, New York USA
%

% Initialize variables to be returned
C = 1;
sigma = 0.3;

% ============================================
% Note: This function returns the optimal C and sigma
%       learning parameters  using the cross validation set.
%        Function "svmPredict" is used to predict the labels on the cross
%         validation set. That is: 
%         predictions = svmPredict(model, Xval);
%         will return the predictions on the cross validation set.
%
%  Note: Prediction  error is calculated using 
%        mean(double(predictions ~= yval))
%
% set up array to store results of experimenting with 
% different values of C and sigma.

results = zeros(64,3);
error_row = 0;            % row counter for loops

for C_test = [0.01 0.03 0.1 0.3 1 3 10 30]
  for sigma_test = [0.01 0.03 0.1 0.3 1 3 10 30]
    error_row = error_row + 1;
   % ++error_row;  %increment row count for results vector
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
   
   svm_prediction = svmPredict(model,Xval);   % generate predicted values vector
    
    prediction_error = mean(double(svm_prediction ~= yval));
    
    results(error_row,:) = [prediction_error C_test sigma_test];
    
  endfor
endfor  

  % Sort rows by first column to select values for "C" and "sigma" with 
  % smallest error.
  
  results_sorted = sortrows(results,1);
  
  C = results_sorted(1,2);
  sigma = results_sorted(1,3);




% =========================================================================

end
