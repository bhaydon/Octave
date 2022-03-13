function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%    COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%   (Bruce Haydon)
%

% Unfold the matrices from "params"
X = reshape(params(1:num_movies*num_features), num_movies, num_features);

Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% Returned variables
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ============================================
% Notes:        Computes the cost function and gradient for collaborative
%               filtering. Cost function is checked first (without 
%               regularization).
%               As a final step, regularization is implemented.

%
% Notes: "X" - (num_movies  x num_features) matrix of movie features
%        "Theta" - (num_users  x num_features) matrix of user features
%        "Y" - (num_movies x num_users matrix) of user ratings of movies
%        "R" - (num_movies x num_users matrix), where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%        "X_grad" - (num_movies x num_features) matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%
%        "Theta_grad" - (num_users x num_features matrix), containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% (nm*nu) = (nm*nf)(nf*nu) - (nm*nu)   ensuring vector math works
  diff=(X*Theta')-Y;
  
  J = sum((diff.^2)(R==1))/2;
  
  % "X_grad"     -> (nm * nf) same size as "X"
  % "Theta_grad" -> (nu * nf) same size as "Theta"
  % In calculating gradients, we take advantage of the fact we have already
  % calculated "diff" in the cost function calculation above.
  % We multiply by "R" to set selected entries not ranked to "0"
  %
  % (nm*nf) =  ((nm*nu)*(nm*nu)) * (nu*nf)
    X_grad  =   (R.*diff) * Theta;
    
  % (nu*nf)    = ((nm*nu)*(nm*nu))' * (nm*nf)
    Theta_grad = (R.*diff)' * X;  
  
  % Now add Cost-function regularization for both "X" and "theta"
  
  J = J + lambda * sum(sum(Theta.^2))/2;  % regularization term of Theta
  J = J + lambda * sum(sum(X.^2))/2;      % regularization term for "X"
  
  
  % Now add regularization to gradients
  %  (nm*nf) = (nm*nf) + (nm*nf)
      X_grad = X_grad + (lambda*X);
      
      Theta_grad = Theta_grad + (lambda*Theta);











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
