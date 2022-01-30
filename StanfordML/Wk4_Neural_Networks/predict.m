function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%   <Bruce Haydon (2022)>

% Useful values: 
% "m" =  number of samples for learning (rows)
% "n" = # feature variables (columns)/(pixels) for each sample  row
m = size(X, 1);    
num_labels = size(Theta2, 1);

%  The following variable "p" will be returned
p = zeros(size(X, 1), 1);

% ==========================================================
% Notes: The following code is used to make predictions using
%               the  learned neural network. "p" is a 
%               vector containing labels between 1 to "num_labels".
%
% Note: The "max" function is used here to return
%       the index of the max element. For more
%       information see 'help max'. Since data is in rows, 
%       use max(A, [], 2) to obtain the max for each row.
%

%insert first column of ones into X for layer 1 bias variable Theta1(0)
%Now each sample row will start with a "1" which will be mult by Theta1(0)
%This command concatenates two matrices together:
%             (a) column of ones with "m" rows  (m x 1)
%             (b) original test data matrix  (m x n)
a1 = [ones(m,1) X];

z2 = (a1*Theta1');

% concatenate a row of ones on top for mult by Theta2(0) with Sigmoid of z2
a2 = [ones(size(z2),1) sigmoid(z2)];
z3 = (a2*Theta2');
a3 = sigmoid(z3);

%Use "max" function to determine which multiclass value fits image best

[maxvalue, maxindexx] = max(a3, [], 2);

% return "p"
p = maxindexx;






% =========================================================================


end
