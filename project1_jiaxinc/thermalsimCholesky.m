function [ Temperature ] = thermalsimCholesky( p, mediumX, mediumY, leftBound, rightBound, topBound, bottomBound )
%THERMALSIMCHOLESKY solves the 2D steady state thermal problem using
%Cholesky factorization
%   INPUT:
%   p:  discretized power density
%   mediumX:    x-dimension of the medium
%   mediumY:    y-dimension of the medium
%	leftBound:	Temperature at the left boundary (x=0), leftBound(j) means
%	the temperature at T(0,j)
%	rightBound:	Temperature at the right boundary (x=N+1)
%	topBound:	Temperature at the top boundary (y=M+1)
%	bottomBound:	Temperature at the bottom boundary (y=0)
%
%   OUTPUT:
%   Temperature: solved thermal map
%
%**************************************************************************
%% Initialize parameters
% Thermal conductivity constant
k = 157;

% Get the number of panels in x-direction and y-diretion
[N, M] = size(p);
len = N * M;

deltaX2 = (N / mediumX) ^ 2;
deltaY2 = (M / mediumY) ^ 2;

%**************************************************************************
%% Compute the coefficients as matrix A 
% Get the diagonal of matrix A for T(i,j) 
A_ij = -2 * (deltaX2 + deltaY2) * ones(1,len);
% Get the coefficients for T(i-1,j) and T(i+1, j)
A_i = deltaX2 * ones(1,len-M);
% Get the coefficients for T(i,j-1) and T(i, j+1)
A_j = deltaY2 * ones(1,len-1);
% Set the boundary condition in matrix A as 0
for i=1:(N-1)
   A_j(i * M)=0;
end

% Combine all the coefficients together to obtain matrix A according to its 
% symmetric and diagonally dominant property.
A = diag(A_ij) + diag(A_i,M) + diag(A_i,-M) + diag(A_j,1) + diag(A_j,-1);

%**************************************************************************
%% Calculate matrix B according to the p
B = -p / k;
% Set the boundary condition to matrix B
B(1,:) = B(1,:) - deltaX2 * leftBound';
B(N,:) = B(N,:) - deltaX2 * rightBound';
B(:,1) = B(:,1) - deltaY2 * bottomBound;
B(:,M) = B(:,M) - deltaY2 * topBound;

% Reshape B to vector
B = reshape(B', len, 1);
save('CholeskyParameter1.mat', 'A', 'B');

%**************************************************************************
%% Implement Cholesky Factorization
% Initialize matrix
L = zeros(len);
V = zeros(len, 1);
X = zeros(len, 1);

% Convert matrix A to positive definite matrix
A = -A;
B = -B;
for i = 1 : len
    % Compute the scaler
    L(i, i) = sqrt(A(i, i) - L(i, :) * L(i, :)');
	for j = i+1 : len
		% Update matrix A
        tmp = A(j, i) - L(j, :) * L(i, :)';
		L(j, i) = tmp / L(i, i);
	end
end

% Solve the lower triangle matrix for V via forward substitution
for i = 1: len-1
   V(i, 1) = B(i) / L(i, i);
   B(i+1, 1) = B(i+1, 1) - L(i+1, 1:i) * V(1:i);
end
V(len, 1) = B(len, 1) / L(len, len);

% Solve the upper triangle matrix for X via backward substitution
U = L';
for i = len : -1 : 2
    X(i, 1) = V(i, 1) / U(i, i);
    V(i-1, 1) = (V(i-1, 1) - U(i-1, i:len) * X(i:len, 1));
end
X(1, 1) = V(1, 1) / U(1, 1);

%**************************************************************************
%% Plot Temperature
Temperature = reshape(X, M, N);
Temperature = Temperature';
thermalplot(Temperature);

save('CholeskyParameter2.mat', 'X', 'Temperature');

end
