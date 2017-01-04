% Generate the recovered lena.bmp fig and calculate the error and median error.
tic;

% Initialize Parameters
blkSize = 16;
numSample = [10 30 50 100 150];
length = size(numSample, 2);
error = zeros(1, length);
errorMedian = zeros(1, length);

% Generate the recovered fig for the 5 numSample respectiely
for i = 1 : length
    imgIn = imgRead('lena.bmp');
    imgOut = imgRecover(imgIn, blkSize, numSample(i));
    imgImproved = medfilt2(imgOut, [3 3]);
    imgShow(imgImproved);
    figName = sprintf('lena_numSample%d.fig', numSample(i));
    saveas(gcf, figName);
        
    % Calculate error and median error
    error(i) = mean(mean((imgOut - imgIn) .^ 2));
    errorMedian(i) = mean(mean((imgImproved - imgIn) .^ 2));   
end

plot(numSample, error, 'r', numSample, errorMedian, 'b');
legend('Error', 'Median Error');
xlabel('Number of Samples');
ylabel('Recovery Error');
axis([10 150 0 8000]);
title('Number of Samples vs. Recovery Error of lena.bmp');

toc;