function [w, infos] = apa_saga(problem, in_options)
% Adaptive proximal average based saga for regularized ERM.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w       solutions of w
%       infos   information

    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options
    local_options = struct('stepsize', 0.01);
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);   
    options = merge_two_options(options, in_options); 
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_baches = floor(n / options.batch_size);
    
    % parameters for the numbers of inner iteration
    s = 0; % s-th iteration of out iteration
    m0 = n;
    mu = 0.8;
    lambda2 = problem.lambda2();
    
    % prepare an array of gradient, and a variable of average gradient.
    grad_array = zeros(num_of_baches, 1);
    grad_ave = 0;
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
       fprintf('apa-saga: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap); 
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        % initialization
        s = s + 1;
        m = m0 / (mu^s);
        stepsize = min(0.1, 10 * mu^s);
        
        for j = 1 : m
            
            % select a min-batch to update
            idx = randi(num_of_baches);
                       
            % calculate gradient
            stored_var = problem.stored_var(w, idx);
            
            % update w use saga gradient
            x_train_i = problem.x_train_i(idx);
            grad = x_train_i * stored_var;
            grad2 = x_train_i * grad_array(idx);
            w = (1 - stepsize * lambda2) * w - stepsize * ( grad_ave + grad - grad2);
            
            % update grad ave
            grad_ave = grad_ave + (grad - grad2) / num_of_baches;
            
            % replace with new grad
            grad_array(idx) = stored_var;
            
            % proximal average
            w = problem.prox_ave(w, stepsize);
            
            % count gradient evaluations
            grad_calc_count = grad_calc_count + 1;
            
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
            fprintf('apa-saga: epoch = %03d, cost = %.16f, stepsize = %.10f, time = %.6f\n', epoch, f_val, stepsize, elapsed_time);
        end
        
        % print result
        if optgap < options.tol_optgap
           fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', option.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('Max epoch reached: max_epoch = %d\n', options.max_epoch);
        end
    end
end

