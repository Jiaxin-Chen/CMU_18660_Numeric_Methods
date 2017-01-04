function [F, G, H] = costFcn(Z, X, y, lambda, t)
% Compute the cost function F(Z)
%
% INPUTS: 
%   Z: Parameter values
%   X(MxN) : trainData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   lambda: Input lambda value
%   t: Initial setPara
%
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To improve the excution speed, please program your code with matrix
% format. It is 30 times faster than the code using the for-loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize parameters
N = size(X, 2);
M = size(X, 1);

W = Z(1 : M);
C = Z(M + 1);
zeta = Z(M + 2 : N + M + 1)';

invt = 1 / t;
term = (W' * X) .* y + C * y + zeta - 1;
term2 = term .^ 2;
y2 = y .^ 2;
yterm = y ./ term;
yterm2 = y ./ term2;
y2term2 = y2 ./ term2;


%% Compute F
F = sum(zeta) + lambda * (W'* W) - invt * sum(log(term)) - invt * sum(log(zeta));


%% Compute Gradient
gradW = 2 * lambda * W - invt * sum(bsxfun(@times, X, yterm), 2);
gradC = -invt * sum(yterm);
gradZeta = 1 - invt * (1 ./ term) - invt * (1 ./ zeta);
G = [gradW; gradC; gradZeta'];


%% Compute (M+1+N) x (M+1+N) Hessian matrix
% M x M HessianW(j)W(k)
HWW = zeros(M, M);  
for i = 1 : M
    temp = bsxfun(@times, X, X(i, :));
    HWW(:, i) = invt * sum(bsxfun(@times, temp, y2term2), 2);
end
HWW = HWW + 2 * lambda * diag(ones(1, M));

% M x 1 HessianW(j)C
HWC = invt * sum(bsxfun(@times, X, y2term2), 2);  

% M x N HessianW(j)zeta(i) 
HWzeta = invt * bsxfun(@times, X, yterm2);  

% 1 x 1 HessianCC
HCC = invt * sum(y2term2); 

% 1 x N HessianCzeta(i)
HCzeta = invt * yterm2;  

% N x N Hessianzeta(i)zeta(i)
Hzetazeta = invt * diag(1./term2 + 1 ./ (zeta.^2));  

% Utilize the symmetric property to construct Hessian matrix
H = [HWW HWC HWzeta; HWC' HCC HCzeta; HWzeta' HCzeta' Hzetazeta];

end