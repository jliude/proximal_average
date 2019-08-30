function [ w, infos ] = admm_svrg(problem, in_options)
% ADMM-SVRG
%
% Input:
%       problem     function(cost/grad/etc)
%       in_options  options
% Output:
%       w       soulutions of w
%       infos   information

    % dimension, samples and groups
    d = problem.dim();
    n = problem.samples();
    
    % establish the matrix
    [C, M] = admm_mat(problem);
%     C = zeros(M, d);
%     l = 1;
%     for i = 1 : K
%         idx = (G{i} - 1)' * M + (l : (l + length(G{i}) - 1));
%         C(idx) = 1;
%         l = l + length(G{i});
%     end
    
    % set local options
    local_options.sub_mode = [];

    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);

    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    stepsize = options.stepsize;
    
    mu = options.mu;
    y = zeros(M, 1);
    u = y;
    % max_inner_iter = 2 * n;
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf("svrg-admm: Epoch = %03d, cost = %.16f, optgap = %.4f\n", epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    gamma = stepsize * mu * norm(C' * C, 2) + 1;
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        % coupute full gradient
        full_grad = problem.full_grad(w);
        grad_calc_count = grad_calc_count + n;
        
        % store vars
        w0 = w;
        
        for j = 1 : n
            
            % update y      
            y = problem.prox_block(C*w + u, 1/mu);
            
            % svrg grad
            idx = randi(n);
            grad = problem.grad(w, idx);
            grad_0 = problem.grad(w0, idx);
            grad_svrg = grad - grad_0 + full_grad;
            grad_calc_count = grad_calc_count + 2;
            
            % update w 
            w = w - (stepsize/gamma) * (grad_svrg + mu * C' * ( C*w - y + u ));
            
            % update u
            u = u + C * w - y;
            
            % store and display infos
            if mod(total_iter, 100) == 0
                
                % measure elapsed time
                elapsed_time = toc(start_time);
                
                % store infos
                [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
                epoch = epoch + 1;
            end
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % display infos
        if options.verbose > 0
            fprintf('svrg-admm: epoch = %03d, cost = %.16f, stepsize = %.5f, optgap = %.10f, time = %.6f\n', epoch, f_val, stepsize, optgap, elapsed_time);
        end
        
        % print result
        if optgap < options.tol_optgap
            fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
        end
    end
end
