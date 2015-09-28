function [seeds,S] = spectral_clustering(X,k)
% Spectral clustering algorithm by Ng, Jordan and Weiss(2002)

% Modified from the code written by Ingo Buerk:
%   Executes the spectral clustering algorithm defined by
%   Type on the adjacency matrix W and returns the k cluster
%   indicator vectors as columns in C.
%
%   'W' - Adjacency matrix, needs to be square
%   'k' - Number of clusters to look for
%
%   References:
%   - Ulrike von Luxburg, "A Tutorial on Spectral Clustering", 
%     Statistics and Computing 17 (4), 2007
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

% © 09/06/2015 Viivi Uurtio, Aalto University
% viivi.uurtio@aalto.fi
%
% This code is for academic purposes only.
% Commercial use is not allowed.

A = gram( X', X', 'gaussian',1); % Gaussian kernel
A(logical(eye(size(A)))) = 0; % diagonal elements = 0

% calculate degree matrix
degs = sum(A, 2);
D    = sparse(1:size(A, 1), 1:size(A, 2), degs);

% compute unnormalized Laplacian
L = D - A;
  
% avoid dividing by zero
degs(degs == 0) = eps;
% calculate D^(-1/2)
D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));

% calculate normalized Laplacian
L = D * L * D;

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
diff   = eps;
[U, ~] = eigs(L, k, diff);

% normalize the eigenvectors row-wise
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));

for i=1:100
    [~,~,~,~,s] = kmedoids(U,k);
    S(i,:)=sort(s); 
end

% for i=1:100
%     centroids = find_centroids(U,k);
%     S(i,:)=sort(centroids);
% end

seeds=mode(S);



end

