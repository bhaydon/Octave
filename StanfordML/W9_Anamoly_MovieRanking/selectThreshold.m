function [bestEpsilon bestF1] = selectThreshold(yval, pval)
% 
%   SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%   outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%
%   'pval' = (m x 1) results 0 < p < 1
%   'yval' = (m x 1) labels: is either '0' or '1'  {0,1}
%            y=1 analmolous example  y=0  normal example
%   
%   ALGORITHM:
%   (a) Model "p(x)" from training set  -> p(x, mu, sigma^2)
%   (a) Fit model "p(x)" on training set {x(1)...x(m))
%   (b) On a cross-validation set example "x", predict:
%  
%            y = 1 if p(x) <  epsilon  {anomaly)
%            y = 0 if p(x) >= epsilon  (normal)
%
bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

[m, n]  = size (yval);

% m = number of training examples in 'yval' and 'pval' (mx1) vectors

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)
    
    % ============================================
    % Notes:        Computes the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: Could have also used predictions = (pval < epsilon) to get a
    %       binary vector of 0's and 1's of the outlier predictions

  % initialize counter variables: tp= true pos, fp = false pos, fn = false neg
  % 'precision' = tp/(tp + fp)
  % 'recall'    = tp/(tp + fn)
  %
  %  if (pval < epsilon) --> anomaly
  
  tp=0;
  fp=0;
  fn=0;

 % for i = 1:m   
 %    if (pval(i) < epsilon)&&(yval(i) == 1)    % true positive
 %      tp=tp + 1;    
 %    elseif (pval(i) < epsilon)&&(yval(i) == 0)  % false positive     
 %         fp = fp + 1;
 %     elseif (pval(i) >= epsilon )&&(yval(i) == 1)  %false negative
 %         fn = fn + 1;       
 %    endif
 
 ;
     
 % true positive = algorithm says  anomaly, yval says anomaly   
   tp = sum((yval==1) & (pval < epsilon));
 

 % false positive = algorithm says  anomaly, yval says normal  
   fp = sum((yval==0) & (pval < epsilon));
 
% false negative = algorithm says  normal, yval says anaomaly   
   fn = sum((yval==1) & (pval >= epsilon));
   
     precision = tp/(tp + fp);
     recall = tp/(tp + fn);
     F1 = (2*precision*recall)/(precision+recall);     

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
endfor

end
