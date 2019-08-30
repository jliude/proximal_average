function [w, infos] = svrg(problem, in_options)
    % Stochastic Variance gradient descent (SVRG) algorithm.
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

    % number of inner iter
    if ~isfield(options, 'max_inner_iter')
        options.max_inner_iter = 2 * num_of_bachces;
    end

    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);

    % display infos
    if options.verbose > 0
        fprintf('svrg: Epoch = %03d, cost = %.16f, optgap = %.12f\n', epoch, f_val, optgap);
    end

    % set start time
    start_time = tic();

    % main loop
    % out loop of svrg
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        % compute full gradient
        full_grad = problem.full_grad(w);
        % store w
        w0 = w;
        grad_calc_count = grad_calc_count + num_of_bachces;

        % inner loop
        for j = 1 : options.max_inner_iter

            % selcet a min-batch to update
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

            % proximal operator
            w = problem.prox_ave(w, stepsize);
            
            % count gradient evalutions
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
        
        % display infos
        if options.verbose > 0
            fprintf('svrg: epoch = %03d, cost = %.16f, optgap = %.10f\n', epoch, f_val, optgap);
        end
    end

    % print results
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end
end

