%INF552_homework2_Kmeans
%@Yiming Liu

load clusters;
X = clusters;
%K: num of clusters
K = 3;
[dataNum,dim] = size(X); 
% Random permutation of all points
R = randperm(dataNum);

% Id:[num*1] matrix which saves the cluster number every point belongs to.
Id = zeros(dataNum, 1);

% Centroid matrix 
Centr = zeros(K, dim);

% Take the first K points in the random permutation as the centriod
for k=1:K
    Centr(k,:) = X(R(k),:);
end

for count = 1:500
    preCentr = Centr;
    fprintf('  Kmeans times %d\n', count)
    % Find the closest point
    for n=1:dataNum
    % Find closest centroid to current point n
        minId = 1; 
        minVal = norm(X(n,:) - Centr(minId,:), 1);
        for j=1:K
            dist = norm(Centr(j,:) - X(n,:), 1);
            if dist < minVal
                minId = j;
                minVal = dist;
            end
        end     
        
    % Assign the point to the closer centroid
        Id(n) = minId;
    end
    
    % Compute centroids
    for k=1:K
        Centr(k, :) = sum(X(Id == k, :));
        Centr(k, :) = Centr(k, :) / length(find(Id == k));
    end
    
    % Check for convergence
    if Centr == preCentr
        break;
    end
 
end
disp('Centroid matrix:');
disp(Centr);

