function optimalLambda = CrossValidation(T, lambda, originalC, numSample)
% Generate optimal lambda by cross validation
% INPUT:
%  T: DCT transformation
%  lambda: the constraints of the L0-Norm regularization
%  originalC: the vector of pixels in a block
%  numSample: how many samples in a block
% OUTPUT:
%  optimalLambda: the best lambda with the minimum error

% Initialize parameters
M = 20;
N = size(lambda, 1);
m = floor(numSample / 6);
error = zeros(N, 1);

sampleIdx = randperm(size(originalC, 1), m);
B = originalC(sampleIdx, :);
A = T(sampleIdx, :);

for t = 1 : M
            
    % Cross validation
    for lambdaIdx = 1 : N                    
        % Randomly select m samples to generate the test set
        testIdx = randperm(size(B, 1), m);
        testSet = B(testIdx, :);
                
        % Generate the train set and calculate DCT coefficients 
        mask = zeros(size(B, 1), 1);
        mask(testIdx) = 1;
        mask = ~ mask;
        trainA = A(mask,:);
        trainB = B(mask,:);                
        trainAlpha = OMP(trainA, trainB, lambda(lambdaIdx));
        
        % Estimate approximation error from the test set
        trainC = T * trainAlpha;
        trainSamSet = trainC(sampleIdx);
        trainSet = trainSamSet(testIdx);                
        diff = norm(testSet - trainSet);
        
        % Calculate error
        temp = 10 * error(lambdaIdx) / (N - 1);
        if (lambdaIdx ~= 1) && (diff > temp)
            error(lambdaIdx) = error(lambdaIdx) + temp;
        else
            error(lambdaIdx) = error(lambdaIdx) + diff;
        end
     end
end
        
% Select lambda with the minimum error
error = error ./ M;
[~,minIdx] = min(error);
optimalLambda = lambda(minIdx);

end

