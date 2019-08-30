function [w, infos] = pa_asgd(problem, in_options)
% Proximal average based accelerated gradient descent method
% for regularized empirical risk minimization.
% "Accelerated Stochastic Gradient Method for Composite Regularization"
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information

    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options
    local_options = [];

    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);   
    options = merge_two_options(options, in_options);  
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    z = w;
    num_of_baches = floor(n / options.batch_size);
    
    % parameters
    alpha = 1;
    gamma = 1;
    
    % store first infos
    clear infos;
    [infos, f_vals, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
       fprintf('Pa-asgd: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_vals, optgap); 
    end    
    
    % start time
    start_time = tic();
    
    % main loop
    count = 1000;
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        for j = 1 : count
           
            % parameter
            L = 10 + 10/gamma + 0.0001/(2 * alpha^2) - 0.0001/alpha;
            alpha = 2/(total_iter + 1);
            gamma = alpha;
            eta = (L + 0.0001/alpha)^(-1);
            %eta = L^(-1);
            
            % update x
            x = ((1 - alpha) * (1 + L*alpha) * w + L * alpha^2 * z) / (1 - alpha + L * alpha);
            
            % select a mini-batch to update
            idx = randi(num_of_baches);
            y = x - eta * problem.grad(w, idx);
            
            % proximal average
            w = problem.prox_ave(y, eta);
            
            % update z
            z = z - (L * (x - w) + 10 * (z -x)) / (L * alpha + 1);
            
            % count gradient evaluations
            grad_calc_count = grad_calc_count + 1;
            
            % store and display infos
            if mod(grad_calc_count, 100) == 0
                
                % measure elapsed time
                elapsed_time = toc(start_time);
                
                % store infos
                [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
                epoch = epoch + 1;
            end
            
            total_iter = total_iter + 1;
        end
        
        % measure elasped time
        elapsed_time = toc(start_time);

        % display infos
        if options.verbose > 0
            fprintf('pa-asgd: epoch = %03d, cost = %.16f, optgap = %.10f, time = %.6f\n', epoch, f_val, optgap, elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

