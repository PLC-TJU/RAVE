function z = expmap(p,X)
%Riemannian Exponential Map
%p:positive definite matrix
%X:symmetric matrix

[u,L] = eig(p);
g = u*sqrt(L);
invg = inv(g);
Y = invg*X*invg';
[v,S] = eig(Y);
gv = g*v;
z = gv*diag(exp(diag(S)))*gv';