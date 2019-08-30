function [w, infos] = apa_svrg(problem, in_options)
% Adaptive proximal average based stochastic variace reduced gradient
% methods for regularized empirical risk minimization.
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
    local_options = struct('stepsize', 0.01);
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);   
    options = merge_two_options(options, in_options);  
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    
    % parameters for the numbers of inner iteration
    s = 0; % s-th iteration of out iteration
    m0 = n;
    mu = 0.8;
    
    % store fisrst infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
        fprintf('APA-SVRG: Epoch = %03d, cost = %.16f, optgap = %.12f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        % compute full gradient
        full_grad = problem.full_grad(w);
        grad_calc_count = grad_calc_count + num_of_bachces;
        
        % store w
        w0 = w;
        
        % initialization
        s = s + 1;
        m = m0 / (mu^s);
        stepsize = min(0.1, 10 * mu^s);
        
        % inner loop
        for j = 1 : m
            
            % select a mini-batch to update
            idx = randi(num_of_bachces);
            
            % get start_index, end_index
            start_index = (idx-1) * options.batch_size + 1;
            if idx < num_of_bachces
                end_index = start_index + options.batch_size - 1;
            else
                end_index = n;
            end
           
            % calculate svrg gradient
            grad = problem.grad(w, start_index:end_index);
            grad_0 = problem.grad(w0, start_index:end_index);
            
            % update w
            w = w - stepsize * (grad - grad_0 + full_grad);
            
            % proximal average
            w = problem.prox_ave(w, stepsize);
            
            % count gradient evaluations
            grad_calc_count = grad_calc_count + 2;
            
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
            fprintf('apa-svrg: epoch = %03d, cost = %.16f, stepsize = %.10f, time = %.6f\n', epoch, f_val, stepsize, elapsed_time);
        end
        
        % print result
        if optgap < options.tol_optgap
           fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', option.tol_optgap);
        elseif epoch == options.max_epoch
            fprintf('Max epoch reached: max_epoch = %d\n', options.max_epoch);
        end
    end
end

