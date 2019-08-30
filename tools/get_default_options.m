function options = get_default_options(n, d)
% Solves' default parameters
    options.stepsizefun     = @stepsize_alg;
    options.step_alg        = 'fix';
    options.step_init       = 0.01;
    options.stepsize        = 0.01;
    options.lambda          = 0.1;    
    options.tol_optgap      = 1.0e-15;
    options.batch_size      = 1;
    options.max_epoch       = 300;
    options.w_init          = zeros(d, 1);
    options.alpha_init      = zeros(n, 1);
    options.f_opt           = -Inf;
    options.permute_on      = 0;
    options.verbose         = true;
    options.store_w         = false;
    options.mu              = 1;
    
    % for nonmooth loss function
    options.tolerate        = options.stepsize;
    options.nonsmooth_point = 1;
    
end
