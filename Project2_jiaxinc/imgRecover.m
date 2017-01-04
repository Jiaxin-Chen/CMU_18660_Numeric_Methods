function imgOut = imgRecover(imgIn, blkSize, numSample)
% Recover the input image from a small size samples
%
% INPUT:
%   imgIn: input image
%   blkSize: block size
%   numSample: how many samples in each block
%
% OUTPUT:
%   imgOut: recovered image
%
% @ 2011 Huapeng Zhou -- huapengz@andrew.cmu.edu


% Initialize parameters
imgOut = zeros(size(imgIn));
lambda = zeros(ceil(numSample / 5), 1);
N = size(lambda, 1);
for i = 1 : (N - 1)
    lambda(i, 1) =  i * 5;
end
lambda(N) = numSample;
row = size(imgIn,1) / blkSize;
col = size(imgIn,2) / blkSize;

% Divide the image into small blocks
for i = 1 : row
    for j = 1 : col
        % Calculate the transformation of DCT
        block = imgIn((i-1) * blkSize+1 : i * blkSize, (j-1) * blkSize+1 : j * blkSize);
        T = DCT(blkSize);
        originalC = reshape(block', blkSize * blkSize, 1);
       
        % Calculate the optimal lambda by cross validation
        optimalLambda = CrossValidation(T, lambda, originalC, numSample);
        
        % Calculate the coefficients of DCT by OMP
        sampleIdx = randperm(size(originalC, 1), numSample);
        B = originalC(sampleIdx, :);
        A = T(sampleIdx, :);
        alpha = OMP(A, B, optimalLambda);
        
        % Recover the blkSize's image by C
        C = T * alpha;
        imgOut((i-1) * blkSize+1 : i * blkSize, (j-1) * blkSize+1 : j * blkSize) = reshape(C, blkSize, blkSize)';

    end
end

end