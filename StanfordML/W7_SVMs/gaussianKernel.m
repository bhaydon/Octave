function sim = gaussianKernel(x1, x2, sigma)
%   RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
%   Bruce Haydon, New York 

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Initialize output variable to zero
sim = 0;

% ============================================
% NOTE :        This function will return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
% Step is deliberately broken into several steps to allow a breakpoint to be 
% placed to determine intermediate results

% diffsq = (x1 - x2).^2;
% diffsqsum = sum(diffsq);
% diffsqsumsig = diffsqsum/(2.*sigma^2);
% sim=exp(-1*diffsqsumsig);

sim = exp(-sum((x1-x2).^2)/(2*sigma^2));


% =============================================================
    
end
