function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. P is set to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 
%  Bruce Haydon (2022)

m = size(X, 1);
num_labels = size(all_theta, 1);

% The following variable is returned
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ===========================================
% Instructions: The following code is used to make predictions using
%               the learned logistic regression parameters (one-vs-all).
%               Sets p to a vector of predictions (from 1 to
%               num_labels).
%
% Note: The "max" function is used to vectorize the result
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
%  test = 5000x401  all_theta=10x401 therefore must use transponse
%  
test=X*all_theta';

sigprob=sigmoid(test);

[maximum, indexx] = max(sigprob,[],2);

p=indexx;



% =========================================================================


end
