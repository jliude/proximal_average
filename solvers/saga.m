function [w, infos] = saga(problem, in_options)
% Stochastic average descent (SAG) algorithm.
% and SAGA algorithm
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information

    % dimentions and samples
    d = problem.dim();
    n = problem.samples();
    
    % define local_options
    local_options = struct('stepsize', 0.0001);
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    stepsize = options.stepsize;
    
    % prepare an array of gradients, and a valiable of average gradient
    grad_array = zeros(num_of_bachces, 1);
    grad_ave = 0;
    lambda2 = problem.lambda2();
    lambda1 = problem.lambda1();

    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf('SAGA: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
                
        for j = 1 : 2000
            
            % selcet a min-batch to update 
            idx = randi(num_of_bachces);

            % calculate gradient
            stored_var = problem.stored_var(w, idx);
            
            % update w use saga gradient
            x_train_i = problem.x_train_i(idx);
            grad = x_train_i * stored_var;
            grad2 = x_train_i * grad_array(idx);
            w = (1 - stepsize * lambda2) * w - stepsize * ( grad_ave + grad - grad2);

            % update grad_ave
            grad_ave = grad_ave + (grad - grad2) / num_of_bachces;
            
            % replace with new grad
            grad_array(idx) = stored_var;
            
            % proximal operator
            w = problem.prox_ave(w, stepsize);
            
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
           fprintf('saga: epoch = %03d, cost = %.16f, stepsize = %.10f, time = %.6f\n', epoch, f_val, stepsize, elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

