function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i
%   <Bruce Haydon, (2022)>
%
% Code will return all classifier parameters in a matrix all_theta (K x (N+1))
% Each row of all_theta correponsds to learned logistic regression parameters
% for one class.
%
% "y" variable is vector of labels from 1-10 where "0" is mapped to label 10
% y=m-dimensional vector of labels where y(j)=0|1, indicates whether jth 
%  training instance belongs to class k (y(j)=1) or if it belongs to a
%  different class (y(j)=0).

% Some useful variables
m = size(X, 1)
n = size(X, 2)

% The following variables will be returned 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Note: theta(:) will return a column vector.
%
%       **Logical Arrays in Octave/MATLAB**
% Note: Use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: fmincg function is used to optimize the cost
%       function. For-loop is used (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Sample Code for usage of fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);


%     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
     
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%      Need to loop through the K classes using count variable "c"   
     
     for c= 1:num_labels;
%     % Set Initial theta to array of zeroes     
      initial_theta = zeros(n+1,1);     
      
      [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
         initial_theta, options); 
         
%      %assign calculated theta for this class to all_theta vector
%      %in cth row    
       all_theta(c,:) = theta';
       
    endfor;




% =========================================================================


end
