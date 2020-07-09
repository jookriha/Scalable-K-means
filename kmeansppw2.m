function C = kmeansppw2(X,k,W)
%
% Weighted K-means++
%
% References: 
%   [1] O. Bachem, L. Mario, and A. Krause. Distributed and provably good
%       seedings for k-means in constant rounds, ICML, 2017.
%   [2] J. Hämäläinen, T. Kärkkäinen, and T. Rossi, Scalable Initialization
%       Methods for Large-Scale Clustering, Arxiv preprint, 2020.


[N,M] = size(X);
Wcs = cumsum(W); C = zeros(k,M);
C(1,:) = X(find(rand < Wcs/Wcs(end),1),:);
D = inf(N,1);
XX = dot(X,X,2);
for ii = 2:k
    D = min(D,bsxfun(@plus,XX,dot(C(ii-1,:),C(ii-1,:),2))-2*(X*C(ii-1,:)'));
    Dcs = cumsum(D.*W);
    Dcsn = Dcs/Dcs(end);
    C(ii,:) = X(find(rand < Dcsn,1),:);
end
[~,L] = min(bsxfun(@plus,-2*(X*C'),sum(C.*C,2)'),[],2);
Lsparse = sparse(L,1:N,1,k,N,N);
Xw = bsxfun(@times,X,W);  L1 = 0;
while any(L ~= L1)
    L1 = L;
    C = bsxfun(@rdivide,Lsparse*Xw,Lsparse*W);
    [~,L] = min(bsxfun(@plus,-2*(X*C'),sum(C.*C,2)'),[],2);
    Lsparse = sparse(L,1:N,1,k,N,N);
end
