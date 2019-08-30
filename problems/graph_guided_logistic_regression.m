function Problem = graph_guided_logistic_regression(x_train, y_train,...
    x_test, y_test, lambda1, lambda2, lambda0, E)
% This file defines the graph-guided logistic regression
% 
% Input:
%       x_train     train data matrix of x of size dxn
%       y_train     train data vector of y of size 1xn
%       x_test      test data matrix of x of size 1xn
%       y_test      test data vector of y of size 1xn
%       lambda1     l1-regularized parameter.
%       lambda2     l2-regularized parameter.
%       lambda0     reguarized parameter of group lasso
%       E           graph matrix, size of dxd
% Output:
%       Problem     problem instance
%
%
% The problem of interest is defined as
%       
%       min f(w) = 1/n sum_i^n f_i(w) + lambda2 |x|^2 + lambda0 sum_E |x_i
%                   - x_j|.
%           where
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i)))

    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);
    [row, col] = find(E == 1);
    K = length(row);
    
    Problem.name = @() 'graph-guided logistic regression';
    Problem.dim = @() d;
    Problem.x = @() x_train;
    Problem.samples = @() n_train;
    Problem.lambda1 = @() lambda1;
    Problem.lambda2 = @() lambda2;
    Problem.lambda0 = @() lambda0;
    Problem.E = @() E;
        
    % cost
    Problem.cost = @cost;
    function f = cost(w)
        % calculate lambda0 sum_E |x_i - x_j|
        w1 = repmat(w, 1, d);
        w2 = repmat(w', d, 1);
        abs_x = abs(w1 - w2);
        sum_E = E .* abs_x;
        sum_E = lambda0 * sum(sum(sum_E));
        
        f = -sum(log(sigmoid(y_train.*(w'*x_train))),2)/n_train + lambda2 * (w'*w) / 2 + sum_E;
    end

    % grad
    Problem.grad = @grad;
    function g = grad(w, indices)
        e = exp(-y_train(indices)' .* (x_train(:,indices)'*w));
        s = e./(1+e);
        g = -(1/length(indices)) * ((s.*y_train(indices)')' * x_train(:,indices)')';
        g = full(g) + lambda2 * w;  
    end
    Problem.full_grad = @(w) grad(w, 1:n_train);

    % stored variable
    Problem.stored_var = @stored_var;
    function r = stored_var(w, i)
       e = exp(-y_train(i)' .* (x_train(:, i)' * w));
       s = e ./ (1 + e);
       r = -s .* y_train(i);
    end

    % get x_train(:, i)
    Problem.x_train_i = @(i) x_train(:, i);

    % block soft thresholding
    Problem.prox_block = @l1_soft_thresh;
    % z: variable t: stepsize
    function v = l1_soft_thresh(z, t)
        v = soft_thresh(z, t * lambda0);
    end

    % problem for graph-guided fused lasso
    Problem.prox_g = @prox_g;
    % z: variable t: stepsize j1, j2: index
    function s = prox_g(z, t, j1, j2)
       s = z;
       minus = sign(z(j1) - z(j2)) * min(t, abs(z(j1) - z(j2))/2);
       s(j1) = z(j1) - minus;
       s(j2) = z(j2) + minus;
    end

    % proximal average
    Problem.prox_ave = @prox_ave;
    % z: variable t: stepsize
    function s = prox_ave(z, t)
        % get nonzero locations
        ss = zeros(d, K);
        for i = 1:K
            ss(:, i) = prox_g(z, K*lambda0*t, row(i), col(i));
        end
        
        s = sum(ss, 2)/K ;
    end

    % prediction results
    Problem.prediction = @prediction;
    function p = prediction(w)
        
        p = w' * x_test;
        
        class1_idx = p>0;
        class2_idx = p<=0;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;        
        
    end

    Problem.accuracy = @accuracy;
    function a = accuracy(y_pred)
        a = sum(y_pred == y_test) / n_test; 
    end
end

