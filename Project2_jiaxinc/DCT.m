function T = DCT(blkSize)
% Generate DCT transformation T
% INPUT:
%  blkSize: block size
% OUTPUT:
%  T: DCT transformation

pixels = blkSize * blkSize;
T = zeros(pixels);

for i = 1 : pixels
    % Calculate x and y
    x = ceil(i / blkSize);
    y = i - (x - 1) * blkSize;

    for j= 1 : pixels
        % Calculate u and v
        u = ceil(j / blkSize);        
        if (u == 1)
            alpha = sqrt(1 / blkSize);
        else
            alpha = sqrt(2 / blkSize);
        end 
        
        v = j - (u - 1) * blkSize;
        if (v == 1)
            belta = sqrt(1 / blkSize);
        else
            belta = sqrt(2 / blkSize);
        end
        
        % Calculate DCT transformation
        cosX = cos(pi * (2 * x - 1) * (u - 1) / (2 * blkSize));
        cosY = cos(pi * (2 * y - 1) * (v - 1) / (2 * blkSize));
        T(i, j) = alpha * belta * cosX * cosY;
    end
end

end