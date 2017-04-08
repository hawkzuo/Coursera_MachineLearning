function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

rc = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
rs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
iters = size(rc,1);
CS = zeros(iters*iters,2);
for i=1:iters
   CS( (i-1)*iters+1:i*iters,1 ) = rc(i) * ones(iters,1);
   CS( (i-1)*iters+1:i*iters,2 ) = rs;
end

C=0;    sigma=0;
MinError = 1;
for i=1:iters*iters
    curmodel= svmTrain(X, y, CS(i,1), @(x1, x2) gaussianKernel(x1, x2, CS(i,2)));
    eY = svmPredict(curmodel,Xval);
    erate_i = sum(eY ~= yval) / size(yval,1);
    if erate_i < MinError
        C=CS(i,1);  sigma = CS(i,2);
        MinError = erate_i;
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
