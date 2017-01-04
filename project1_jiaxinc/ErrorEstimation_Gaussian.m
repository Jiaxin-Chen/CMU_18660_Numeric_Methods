% This script is used for running and estimating the error of thermalsimGauss.m

% Choose the case to load
load('case1');

% Call thermalsimGauss.m to calculate the temperauture and plot the
% distribution as well as recording the elasped time
tic;
Temperature = thermalsimGauss( p, mediumX, mediumY, leftBound, rightBound, topBound, bottomBound );
toc;

% Load the Gaussian Parameter(matrix A, matrix B, vector X and Temperature 
load('GaussianParameter.mat');

% Calculate the golden solution by \ in MATLAB
golden_solution = A \ B;

% Calculate the error
error = sqrt((sum(sum(golden_solution - X).^2)) / sum(sum(golden_solution.^2)));
fprintf('The error:');
disp(error);


