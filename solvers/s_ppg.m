function [w, infos] = s_ppg( problem, in_options )
% Stochastic proximal proximal gradient algorithm
%
% Input:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w       solution of w
%       infos   information

    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % define local_options
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);  
    options = merge_two_options(options, in_options);
    
    % initialized
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    stepsize = options.stepsize;
    
    % prepare an array of z, and average z
    z_array = zeros(d, num_of_bachces);
    z_ave = mean(z_array, 2);
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf('S_ppg: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        for j = 1 : num_of_bachces
           
            % step 1
            w = problem.prox(z_ave, stepsize);
            
            % select a coordinate to update
            idx = randi(num_of_bachces);
            
            % proximal of loss
            y = problem.prox_g(2 * w - z_array(:, idx), stepsize, idx);
            
            % update z_array and  z_ave
            temp = z_array(:, idx);
            z_array(:, idx) = temp + y - w;
            z_ave = z_ave + (z_array(:, idx) - temp) / num_of_bachces;
            
            total_iter = total_iter + 1;
        end
        
        % measure elasped time
        elapsed_time = toc(start_time);
         
        % count gradient evalution
        grad_calc_count = grad_calc_count + num_of_bachces;
        epoch = epoch + 1;
        
                
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           
            
        % display infos
        if options.verbose > 0
            fprintf('S-ppg: PASS_GRAD = %d, cost = %.16f, optgap = %.10f, time = %.6f\n', grad_calc_count/num_of_bachces, f_val, optgap, elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);  
    end
    
end

