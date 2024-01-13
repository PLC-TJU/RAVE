function Z = logmap(p,x)
%Riemannian Log Map
%p,x:positive definite matrix

[u,L] = eig(p);
g = u*sqrt(L);
invg = inv(g);
y = invg*x*invg';
[v,S] = eig(y);
gv = g*v;
Z = gv*diag(log(diag(S)))*gv';