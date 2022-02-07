function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
% Bruce Haydon (#bhaydon) 2022  https://linkedin.com/in/bhaydon
%
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   "nn_params" and need to be converted back into the weight matrices. 
%   "num_labels" = output layer size
% 
%   The returned parameter grad will be an "unrolled" vector of the
%   partial derivatives of the neural network.
%

%  This code reshapes "nn_params" back into the parameters Theta1 and Theta2, 
%  the weight matrices for our 2 layer neural network
%
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Theta1 = 25 x 401  (between input (s=400) and hidden (s=25) layer)
% Theta2 = 10 x 26   (between hidden (s=25) and output (s=10) layer)
                 
% Setup some useful variables - "m" = number of training rows
m = size(X, 1);
         
% The following variable are returned: 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== DESCRIPTION ======================
%
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implements backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Returns partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. Calculated gradients are checked
%         by running checkNNGradients procedure.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. Vector needs to be mapped into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Note: Backprop is implemented using  a for-loop
%               over the training examples, and not vector math.
%
% Part 3: Implements regularization with the cost function and gradients.
%
%         Note: Implemented using algorithm on
%               backpropagation. Gradients are calculated for
%               the regularization separately and then added to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part: Feedforward 
%
% Each integer (1-10) converted to a 10-row column vector "Y"
% with a "1" at the row of the integer's value, zeroes at all other rows.
% Therefore, 5000 values of "y" requires "Y" be a 10x5000 matrix
% We will use individual rows of a created "identity matrix" "Ident" 
% (diagonal row of ones) - transposed from rows to columns - to create this
% new matrix "Y" using the following logic.

Y=zeros(num_labels, m);

% Create identity matrix 10x10 - diagonal 1's 
Ident = eye(num_labels);

% Assign each column in "Y" the integer value of corresponding value in "y"
% with a "1" in the corresponding row in "Y" (rows 1-10)

for i = 1:m;
  Y(:,i) = Ident(:, y(i));
end

% Feedforward logic begins here
% Create matrix for "a1" = X (input layer)

a1 = X;

% Now insert column of ones at left - a1(0) - to multiply with theta1(zero)
column_of_ones = ones(m,1);

% append this column of "1s" to existing "X" matrix
a1 = [column_of_ones X];  % append 1's column to "X" as first column 
z2 = a1 * Theta1';

a2 = sigmoid(z2);
column_of_ones = ones(size(z2,1), 1);
a2 = [column_of_ones a2];

% (5000x26)*(26x10) --> (5000x10) z3 matrix
z3 = a2 * Theta2';  
a3 = sigmoid(z3);
h=a3;

% from logistic regression: J= (1/m) * (-Y*log(h) - (1 - Y) * log(1-h))

% J1 = (-Y)' .* log(h)  -  (1-Y)' .* log(1-h)
% J2= sum (  (-Y)' .* log(h)  -  (1-Y)' .* log(1-h) )
% J3 = (1/m) * sum ( sum (  (-Y)' .* log(h)  -  (1-Y)' .* log(1-h) ));


 J = (1/m) * sum ( sum (  (-Y)' .* log(h)  -  (1-Y)' .* log(1-h) ));

% Note that we should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of 
% each matrix. Remove first columm on each matrix

Theta1_skinny = Theta1(:,2:size(Theta1,2));

Theta2_skinny = Theta2(:,2:size(Theta2,2));

%Progressively sum regularization component to allow inspection
% of totals at each stage

Regularization = (sum(sum((Theta1_skinny .^2))) + sum(sum((Theta2_skinny .^2))));
Regularization = Regularization * lambda/(2*m);

% Add regularization component to existing cost

J = J + Regularization;


%=====================BACK PROP=======================================

% Append column of ones to "X" as bias value
X = [ones(m,1) X];

for i=1:m
  
  %% Step 1 - Forward Prop
  %
  % "a1" is ith row of "X" --> (1x401), Theta1 => (25x401)
  % "z2" --> (1x25)   single row of 25 columns
  a1 = X(i,:);
  
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];           % add bias column to left value = '1'
  
  % a2 -> (1x26)  Theta2 -> (10x26)
  % z3, a3 -> (1x10)
  z3 = a2 * Theta2';
  a3 = sigmoid (z3);   %final activation layer
  
  h=a3;
  
  %% Step 2 - Backward Prop - Calc delta at output layer
  % a3->(1x10)  Y->(10x5000)  Y(:,i)->(10x1) Y(:,i)' ->(1x10)
  % delta_3 -> (1x10)
  delta_3 = a3 - Y(:,i)';
  
  
  %% Step 3 - Backward Prop - Calc delta for hidden layer
  % Theta2 ->(10x26) delta_3->(1x10)
  % delta_3*Theta2 --> (1x10)*(10x26) -> (1x26)
  % z2 modified -> (1x26)
  % delta_2 -> (1x26)
  z2=[1 z2]; % bias
  
 % delta_3->(1x10) Theta2->(10x26) z2->(1x26) 
 % delta_2 should be (1x25) 
 % Note use of dot product in formula
  
  delta_2 = ( delta_3 * Theta2) .* sigmoidGradient(z2);
  
  
  %% Step 4 - Gradient Accumulation
  % Accumulate the gradient. 
  % Note that delta_2(0) should be removed which is next step
  delta_2 = delta_2(2:end);
  
  % Theta2_grad->(10x26) delta_3->(1x10) a2->(1x26)
  % size(delta_3)     %(1x10)
  % size(a2)          %(1x26)
  % size(Theta2_grad) %(10x26)
  
  Theta2_grad = Theta2_grad + (delta_3' * a2);
  
  % size(delta_2)     %(1x25)
  % size(a1)          %(1x401)
  % size(Theta1_grad) %(25x401)
  
	Theta1_grad = Theta1_grad + (delta_2' * a1);
  
   %% Step 5 - Obtain unregularized gradient
   %  Obtain the (unregularized) gradient for the neural network 
   %  cost function by dividing the accumulated gradients by 1/m
   %
   
  Theta1_grad = Theta1_grad ./ m;
  Theta2_grad = Theta2_grad ./ m;
  
  

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
