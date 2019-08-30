function [C, M] = admm_mat(problem)
% To establish the correlation matrix for admm algorithms
%
% Iuput:
%       problem: functions(cost/grad/etc)
% Output:
%       C: the correlation matrix for admm algorithm.
%       M: dimensions of y.

    d = problem.dim();
    %% overlapping group lasso
    if strcmp(problem.name(), 'overlapping_group_lasso_regression')
        G = problem.G();
        K = problem.group();
        M = problem.total(G);
        
        % establish the matrix
        C = zeros(M, d);
        l = 1;
        for i = 1 : K
            idx = (G{i} - 1)' * M + (l : (l + length(G{i}) - 1));
            C(idx) = 1;
            l = l + length(G{i}); 
        end
        
    %% graph-guided fused lasso
    elseif strcmp(problem.name(), 'graph-guided logistic regression')
            E = problem.E();
            [row, col] = find(E == 1);
            M = length(row);
            
            % establish the matrix
            C = zeros(M, d);
            for i = 1 : M
                C(i, row(i)) = 1;
                C(i, col(i)) = -1;
            end
    else
        error('Wrong!');
    end
    
end

