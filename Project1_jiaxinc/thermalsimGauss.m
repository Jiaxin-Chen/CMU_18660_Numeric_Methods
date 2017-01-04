function [ Temperature ] = thermalsimGauss( p, mediumX, mediumY, leftBound, rightBound, topBound, bottomBound )
%THERMALSIMGAUSS solves the 2D steady state thermal problem using Gaussian
%elimination
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

deltaX2 = ((N) / mediumX) ^ 2;
deltaY2 = ((M) / mediumY) ^ 2;

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
B = - p / k;
% Set the boundary condition to matrix B
B(1,:) = B(1,:) - deltaX2 * leftBound';
B(N,:) = B(N,:) - deltaX2 * rightBound';
B(:,1) = B(:,1) - deltaY2 * bottomBound;
B(:,M) = B(:,M) - deltaY2 * topBound;

% Reshape B to vector
B = reshape(B', len, 1);

%**************************************************************************
%% Construct the linear system
GE = zeros(len, len + 1);
GE(:, 1:end-1) = A;
GE(:, end) = B;

%**************************************************************************
%% Implement Gaussian Elimination
% Convert matrix A in GE to an upper triangular matrix
for i = 1 : len+1 
    for j = i+1 : len 
        % Calculate the factor and replace the old GE with updated one
        factor = GE(i, :) .* (-GE(j, i)/GE(i, i));
        GE(j, :) = factor + GE(j, :); 
    end 
end

% Solve the upper triangle matrix for X via backward substitution
X = zeros(len, 1);
for i = len : -1 : 2
    X(i, 1) = GE(i, len+1) / GE(i, i);
    GE(i-1, len+1) = GE(i-1, len+1) - GE(i-1, i:len) * X(i:len);
end
X(1, 1) = GE(1, len+1) / GE(1, 1);

%**************************************************************************
%% Plot Temperature

Temperature = reshape(X, M, N);
Temperature = Temperature';
thermalplot(Temperature);

save('GaussianParameter.mat', 'A', 'B', 'X', 'Temperature');

end


