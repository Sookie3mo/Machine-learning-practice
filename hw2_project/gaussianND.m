%INF552_homework2
%@Yiming Liu
%GAUSSIANND 
function [ result ] = gaussianND(X, mea, sigma)

%  mea - Row vector for the mean; sigma - Covariance matrix.

lengh = size(X, 2);

%Calculate every row's mean
meanDiff = bsxfun(@minus, X, mea);

%Calculate the multivariate gaussian

result = 1 / sqrt((2*pi)^lengh * det(sigma)) * exp(-1/2 * sum((meanDiff * inv(sigma) .* meanDiff), 2));

end