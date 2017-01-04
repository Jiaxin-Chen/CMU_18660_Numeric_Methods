tic;

% Choose the mat file to run
load('feaSubEImg.mat');
%load('feaSubEOvert.mat');

% Initialize parameters
sixFold = 6;
classA = class{1};
classB = class{2};
dataNum = size(classA, 2);
feaNum = size(classB, 1);
stepSize = dataNum / sixFold;
labelA = ones(1, dataNum);
labelB = -labelA;

Ac = zeros(sixFold, 1);
optLambda = zeros(sixFold, 1);
W = cell(1, sixFold);
C = cell(1, sixFold);

setPara.t = 1;
setPara.beta = 15;
setPara.Tmax = 1000000;
setPara.tol = 0.000001;
setPara.W = ones(feaNum, 1);
setPara.C = 0;

% Implement 1st-level six-fold cross validation
for i = 1 : sixFold
    t = setPara.t;
    beta = setPara.beta;
    Tmax = setPara.Tmax;
    tol = setPara.tol;
    W{i} = setPara.W;
    C{i} = setPara.C;
       
    % Generate trainData and testData
    testIdx = (i-1) * stepSize + 1 : i * stepSize;
    testData = [classA(:, testIdx) classB(:, testIdx)];
    testLabel = [labelA(:, testIdx) labelB(:, testIdx)];
    trainIdx = setdiff(1 : dataNum, testIdx);
    trainData = [classA(:, trainIdx) classB(:, trainIdx)];
    trainLabel = [labelA(:, trainIdx) labelB(:, trainIdx)];
    
    % Calculate the optimal lambda of each fold
    optLambda(i) = getOptLamda(trainData, trainLabel, setPara);
    
    % Initial guess of each fold
    trainNum = size(trainData, 2);
    zeta = zeros(1, trainNum);
    for k = 1 : trainNum
        zeta(1, k) = max(1-trainLabel(k)*(W{i}'*trainData(:, k)+ C{i}), 0) + 0.001;
    end
    init_Z = [W{i}; C{i}; zeta'];

    % Implement interior point method
    while (t <= Tmax)
        [optSolution, err] = solveOptProb_NM(@costFcn, init_Z,tol, trainData, trainLabel, optLambda(i), t);
        init_Z = optSolution;
        t = t * beta;
    end
    
    % Calculate the test accuracy of each fold
    W{i} = optSolution(1 : feaNum);
    C{i} = optSolution(feaNum + 1);
    predict = (W{i}' * testData + C{i}) .* testLabel;

    Ac(i, 1) = sum(predict > 0) / size(testData, 2);      
end

% Generate mean accuracy and standard deviation of six-fold
meanAc = mean(Ac);
stdAc = std(Ac);

% Plot the channel weights for the first training fold
show_chanWeights(abs(W{1}));
title('Channel Weights Plot');

toc;
