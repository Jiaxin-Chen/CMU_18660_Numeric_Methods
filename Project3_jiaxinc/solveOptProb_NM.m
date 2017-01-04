function [optSolution, err] = solveOptProb_NM(costFcn,init_Z,tol, X, y, lambda, t)
% Compute the optimal solution using Newton method
%
% INPUTS:
%   costFcn: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   lambda: Input lambda value
%   t: Initial setPara
%
% OUTPUTS:
%   optSolution: Optimal soultion
%   err: Errorr
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

% Initialize parameters
dataNum = size(X, 2);
feaNum = size(X, 1);

Z = init_Z;

[F, G, H] = feval(costFcn, Z, X, y, lambda, t);
deltaZ = -H \ G;
err = -(G' * deltaZ);


% Set the error 2*tol to make sure the loop runs at least once
while (err/2) > tol
   
    % Implement backtracking line search
    s = 1;
    while(true)
       updateZ = Z + s * deltaZ;
       W = updateZ(1 : feaNum);
       C = updateZ(feaNum + 1);
       zeta = updateZ(feaNum + 2 : feaNum + dataNum + 1)';
       
       % Calculate the stopping criterion
       term = (W' * X) .* y + C * y + zeta - 1;
       index1 = find(term <= 0, 1);
       index2 = find((zeta <= 0), 1);
       
       if (isempty(index1) && isempty(index2))
           break;
       else
           s = 0.5 * s;
       end       
    end
    
    Z = Z + s * deltaZ;
    
    % Execute the cost function at the current iteration
    % F : function value, G : gradient, H, hessian
    [F, G, H] = feval(costFcn, Z, X, y, lambda, t);

    deltaZ = -H \ G;
    err = -(G' * deltaZ);
    
end

optSolution = Z;
    
end


