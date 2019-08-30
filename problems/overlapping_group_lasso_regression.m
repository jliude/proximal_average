function [Problem] = overlapping_group_lasso_regression(x_train, y_train,...
    x_test, y_test, lambda1, lambda2, lambda0, G)
% This file defines the OVERLAPPING GROUP LASSO REGRESSION
% 
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda1     l1-regularized parameter.
%       lambda2     l2-regularized parameter.
%       lambda0     reguarized parameter of group lasso
%       G           group infomations. cell structure
% Output:
%       Problem     problem instance.
%
% 
% The problem of interest is defined as
%
%       min f(w) = 1/n sum_i^n f_i(w) + lambda1 |w|_1 + lambda2/2 |w|^2 +
%                  1/G lambda0 sum_k^G |w_g_k|
%           where
%           f_i(w) = 1/2 * (w' * x_i - y_i)^2
%
% 'w' is the model parameter of size d verctor

    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);
    K = length(G);    
    
    Problem.name = @() 'overlapping_group_lasso_regression';
    Problem.dim = @() d;
    Problem.x = @() x_train;
    Problem.samples = @() n_train;
    Problem.lambda1 = @() lambda1;
    Problem.lambda2 = @() lambda2;
    Problem.lambda0 = @() lambda0;
    Problem.G = @() G;
    Problem.group = @() K;
        
    % size
    Problem.total = @total;
    function M = total(G)
       M = 0;
       for k = 1 : length(G)
           M = M+ length(G{k});
       end   
    end

    % cost
    Problem.cost = @cost;
    function f = cost(w)
        % record norm
        vec = zeros(K, 1);
        
        for i = 1:K
            idx = G{i};
            vec(i) = norm(w(idx));
        end
        
        residual = x_train * w - y_train;
        f = (residual' * residual)/ (2 * n_train) + lambda2/2 *(w'*w)...
            + lambda1 * norm(w, 1) + lambda0 * sum(vec);
    end

    % residual
    Problem.res = @res;
    function r = res(w, indices)
        r = x_train(indices, :) * w - y_train(indices);
    end

    % stored variable
    Problem.stored_var = @res;
    
    % get x_train(:, i)
    Problem.x_train_i = @(i) x_train(i, :)';
    
    % grad for f_i(w) + lambda2/2 |w|^2
    Problem.grad = @grad;
    function g = grad(w, indices)
        residual = x_train(indices, :) * w - y_train(indices);
        g = x_train(indices, :)' * residual / length(indices) + lambda2*w;
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)
        g = grad(w, 1:n_train);
    end

    Problem.ind_grad = @ind_grad;
    function g = ind_grad(w, indices)
        residual = x_train(indices, :) * w - y_train(indices);
        g = x_train(:,indices) * diag(residual) + lambda2 * repmat(w, [1 length(indices)]);
    end

    % prox for l1
    Problem.prox = @(w, t) soft_thresh(w, t*lambda1);
    
    % hinge
    Problem.hinge = @hinge;
    % z: variable t: stepsize j: index
    function h = hinge(z, t, j)
        idx = G{j};
        h = max(0, 1 - t/norm(z(idx)));
    end

    % block soft thresholding
    Problem.prox_block = @prox_block;
    % z: variable t: stepsise
    function h = prox_block(z, t)
        h = zeros(length(z), 1);
        for i = 1:length(G)
            idx = G{i};
            h(idx) = max(0, 1 - lambda0*t/norm(z(idx))) * z(idx);
        end
    end
            
    % problem for overlapping group lasso
    Problem.prox_g = @prox_g;
    % z: variable t: stepsize j: index
    function s = prox_g(z, t, j)
        idx = G{j};
        s = z;
        s(idx) = max(0, 1 - t/norm(z(idx))) * z(idx);
    end

    % proximal average
    Problem.prox_ave = @prox_ave;
    % z: variable t: stepsize
    function s = prox_ave(z, t)
        vec = zeros(d, K);
        
        for i = 1:K
           vec(:, i) = prox_g(z, K*lambda0*t, i) ;
        end
        s = sum(vec, 2)/K;
    end
    
    Problem.prediction = @prediction;
    function p = prediction(w)
       p = w' * x_test; 
    end

    Problem.mse = @mse;
    function e = mse(y_pred)
        e = sum((y_pred-y_test).^2)/ (2 * n_test);
    end
end

