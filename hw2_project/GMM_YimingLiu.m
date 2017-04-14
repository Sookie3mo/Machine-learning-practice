%INF552_homework2_GMM
%@Yiming Liu

load clusters
X = clusters;

num = size(X,1);     %size of dataset : 150
k = 3;               %num of clusters
dim = 2;            %dimension of dataset

%choose k points randomly as initial means

buf = randperm(num);
mea = X(buf(1:k), :);

% Initial variance of each cluster
for j = 1:k
    sigma{j} = cov(X);
end

%Probabilities of each cluster; ones(M,N): M-by-N matrix of ones.
phi = (1 / k) * ones(1, k);

%W saves the probability that each point belongs to each cluster.
% row--data point,col--cluster
W = zeros(num, k);

%==========================E-STEP=======================================
% Loop until convergence
for loop = 1:3000
   fprintf('  EM times %d\n', loop);
    prob = zeros(num, k);

% Evaluate the Gaussian for all data points for cluster 'j'.
    for j = 1 : k
        prob(:, j) = gaussianND(X, mea(j, :), sigma{j});
    end
    
    % prob_w = prob * phi
    % prob [num x k]  phi [1 x k] prob_w [num x k]  
    %W : W(j)
    
    %bsxfun : Binary Singleton Expansion Function
    %  @times      Array multiply
    % @rdivide     Right array divide
    prob_w = bsxfun(@times, prob, phi);
    W = bsxfun(@rdivide, prob_w, sum(prob_w, 2));

    %=========================M-STEP======================================
    %Hold the previous mean for checking convergence
    preMea = mea; 
    
    % [1] Phi --> adapted mixture phi(weight)
    for j = 1 : k
        phi(j) = mean(W(:, j), 1);    
      
    %[2] Mean --> adapted mixture mean
        mea(j, :) = weightedAverage(W(:, j), X);
        mu = bsxfun(@minus, X, mea(j, :));
       
    %[3] Covariance matrix --> adapted mixture cov matrix
        sigma_k = zeros(dim, dim);
        for i = 1 : num
            sigma_k = sigma_k + (W(i, j) .* (mu(i, :)' * mu(i, :)));
        end
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
    % Stop the loop if convergence
    if (mea == preMea)
        break
    end
end

 %=======Show mean, amplitude, convariance matrix=================
    disp('Mean:');
    disp(mea);
    
    disp('Convariance matrix:');
    for n = 1:k
        disp(sigma{n});
    end
  
   In = zeros(num);
   for i = 1:num
       Wrow = W(i, :);
       [M,I] = max(Wrow);
       In(i) = I;
   end
   a = 0;b = 0; c = 0;
  for i = 1:num
      if In(i) == 1 
          a = a + 1;
      elseif In(i) == 2
          b = b + 1;
      else In(i) == 3;
          c = c + 1;
      end
  end
  Amp1 = a/num; Amp2 = b/num; Amp3 = c/num;
disp('Amplitude:');
Amp = [Amp1, Amp2, Amp3];
disp(Amp);