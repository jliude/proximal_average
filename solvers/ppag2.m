function [ w, infos ] = ppag2(problem, in_options)
% proximal proximal saga (low storage)
%
% Input:
%       problem     function(cost/grad/etc)
%       in_options  options
% Output:
%       w       solutions of w
%       infos   informantion

    % dimension, samplkes and groups
    d = problem.dim();
    n = problem.samples();
    K = problem.group();
    G = problem.G();

    % set local options
    local_options.sub_mode = [];

    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);

    % intialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    u = zeros(d, 1);
    stepsize = options.stepsize;
                                                                                                      
    % lambda1 = problem.lambda1();
    lambda2 = problem.lambda2();
    lambda0 = problem.lambda0();

    % prepare an array
    grad_array = zeros(1, n);
    map_array = zeros(d, K);
    grad_ave = 0;
    map_ave = mean(map_array, 2);

    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);

    % display infos
    if options.verbose == true
        fprintf("ppag: Epoch = %d, cost = %.16f, optgap = %.4f\n", epoch, f_val, optgap);
    end

    % set start time
    start_time = tic();

    % main loop
    while epoch < options.max_epoch
        for j = 1:n*10

            % select two indexes
            idx_1 = randi(n); % for stochastic grad
            idx_2 = randi(K); % for stochastic prox
            x = problem.x_train_i(idx_1);

            % get stored residual and hinge
            res_s = grad_array(idx_1);

            % get old grad and prox
            grad_s = x * res_s + lambda2 * u;
            map_s = map_array(:, idx_2);

            % update z
            z = w + stepsize * (map_s - map_ave + grad_s - grad_ave);

            % calc current u
            u = problem.prox_g(z, stepsize*lambda0, idx_2);
            
            % calc map
            map = (z - u) / stepsize;
 
            % calc current residual grad
            res = problem.res(u, idx_1);
            grad = x * res + lambda2 * u;

            % update y x
            y = u - stepsize*grad;
            w =  problem.prox(y, stepsize);

            % update grad_ave prox_ave
            grad_ave = grad_ave + (grad - grad_s) / n;
            map_ave = map_ave + (map - map_s) / K;

            % replace with new residual hinge
            grad_array(idx_1) = res;
            map_array(:, idx_2) = map;

            total_iter = total_iter + 1;
        end

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluation
        grad_calc_count = grad_calc_count + n;
        epoch = epoch + 1;

        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           
        
        % display infos
        if options.verbose == true
            fprintf("ppag: Epoch = %03d, cost = %.16f, optgap = %.4f\n", epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

