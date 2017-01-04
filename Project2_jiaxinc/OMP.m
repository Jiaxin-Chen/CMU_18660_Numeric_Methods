function alpha = OMP(A, B, lambda)
% Utilize Orthogonal Matching Pursuit(OMP) to solve the optimization with 
% L0-Norm Regularization.
% INPUT:
%  A: sampled transformation
%  B: sampled vector of pixels
%  lambda: the constraints of the L0-Norm Regularization
% OUTPUT:
%  alpha: the sparse solution of the L0-Norm Regularization

% Initialize parameters
threshold = 4;
F = B;
col = size(A, 2);
omega = zeros(col, 1);
alpha = zeros(col, 1);
length = size(omega, 1);

for p = 1 : lambda
    % Calculate maximum A*F as the largest inner product value
    [~, indexTheta] = max(abs(A' * F));
    
    % Update omega
    omega(indexTheta, :) = indexTheta;

    % Least squares
    updateA = [];
    for i = 1 : length
        if(omega(i, 1) ~= 0)
            updateA = [updateA A(:, i)];
        end
    end
    updateAlpha = updateA \ B;
    
    % Update F 
    F = B;
    index = 1;
    for i = 1 : length
       if(omega(i, 1) == 0)         
           alpha(i, 1) = 0;
       else
           alpha(i, 1) = updateAlpha(index, 1);
           F = F - alpha(i, 1) * A(:, i);
           index = index + 1;
       end
    end  
    
    % Set threshold to reduce meaningless computation
    if (norm(F) < threshold)
        break;
    end
end
end