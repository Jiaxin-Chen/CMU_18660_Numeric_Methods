function optLamda = getOptLamda(X, Y, setPara)
% Get the optimal lamda
%
% INPUTS:
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.beta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiLamda: Optimal lamda value 
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

%% Initialize parameters
t = setPara.t;
beta = setPara.beta;
Tmax = setPara.Tmax;
tol = setPara.tol;
W = setPara.W;
C = setPara.C;

fiveFold = 5;
lambda = [0.01, 1, 100, 10000];

% Expand lambda vector size by increased step size
% lambdaNum = 10;
% lambda = zeros(lambdaNum, 1);
% lambda(1) = 0.01;
% for i = 2 : (lambdaNum-1)
%     lambda(i) = lambda(i-1) * 5;
% end
% lambda(lambdaNum) = 10000;

lambdaNum = length(lambda);
accuracy = zeros(lambdaNum, 1);

dataNum = size(X, 2) / 2;      % 100
feaNum = size(X, 1);           % 204
stepSize = dataNum / fiveFold; % 20

classA = X(:, 1 : dataNum);
classB = X(:, dataNum + 1 : end);
labelA = Y(1 : dataNum);
labelB = Y(dataNum + 1 : end);


%% Generate optimal lambda
for i = 1 : lambdaNum  
    Ac = zeros(5, 1);

    % Implement 2nd-level five-fold cross validation
    for j = 1 : fiveFold
        testIdx = (j-1)*stepSize+1 : j*stepSize;
        testData = [classA(:, testIdx) classB(:, testIdx)];
        testLabel = [labelA(:, testIdx) labelB(:, testIdx)];
        trainIdx = setdiff(1 : dataNum, testIdx);
        trainData = [classA(:, trainIdx) classB(:, trainIdx)];
        trainLabel = [labelA(:, trainIdx) labelB(:, trainIdx)];
    
        % Initial guess
        trainNum = size(trainData, 2);
        zeta = zeros(1, trainNum);
        for k = 1 : trainNum
            zeta(1, k) = max(1-trainLabel(k)*(W'*trainData(:, k)+ C), 0) + 0.001;
        end        
        init_Z = [W; C; zeta'];
        
        % Implement interior point method
        t = setPara.t;        
        while (t <= Tmax)
            [updateZ, err] = solveOptProb_NM(@costFcn, init_Z,tol, trainData, trainLabel, lambda(i), t);
            init_Z = updateZ;
            t = t * beta;
        end
        
        % Calculate accuracy of each fold
        updateW = updateZ(1 : feaNum);
        updateC = updateZ(feaNum + 1);
        predict = (updateW' * testData + updateC) .* testLabel;
        
        Ac(j, 1) = sum(predict > 0) / size(testData, 2);
        
    end
    accuracy(i) = mean(Ac);
  
end

%% Sort the accuracy of five-fold to find the optimal lambda
[~, index] = max(accuracy);
optLamda = lambda(index(1));

end