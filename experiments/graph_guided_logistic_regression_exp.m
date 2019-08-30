% the experiments for graph-guided logistic regression
clc;
clear all;

% select a dataset
disp('======= 1: a9a, 2: german.numer, 3: w7a, 4: cod-rna ========')
dataset = input('Please select a dataset: ');

% 
switch(dataset)
    case 1
        % get the inverse covariance matrix
        E = csvread('D:\Proximal average\R\datas\a9a.csv', 1, 1);
        E(abs(E) > 0.5) = 1;
        E(abs(E) == 0.5) = 1;
        E(abs(E) < 0.5) = 0;
        
        % define the problem
        [y_train, x_train] = libsvmread('D:\Proximal average\data\graph_guided_logistic_regression\a9a');
        x_train = x_train';
        y_train = y_train';
        lambda0 = 1e-4; lambda2 = 1e-4; lambda1 = 0;
        
    case 2
        % get the inverse covariance matrix
        E = csvread('D:\Proximal average\R\datas\german_numer.csv', 1, 1);
        E(abs(E) > 0.5) = 1;
        E(abs(E) == 0.5) = 1;
        E(abs(E) < 0.5) = 0;
        
        % define the problem
        [y_train, x_train] = libsvmread('D:\Proximal average\data\graph_guided_logistic_regression\german.numer_scale');
        x_train = x_train';
        y_train = y_train';
        lambda0 = 1e-3; lambda2 = 1e-3; lambda1 = 0;
        
    case 3
        % get the inverse covariance matrix
        E = csvread('D:\Proximal average\R\datas\w7a.csv', 1, 1);
        E(abs(E) > 0.5) = 1;
        E(abs(E) == 0.5) = 1;
        E(abs(E) < 0.5) = 0;
        
        % define the problem
        [y_train, x_train] = libsvmread('D:\Proximal average\data\graph_guided_logistic_regression\w7a');
        x_train = x_train';
        y_train = y_train';
        lambda0 = 1e-4; lambda2 = 1e-4; lambda1 = 0;
        
    case 4
        % get the inverse covariance matrix
        E = csvread('D:\Proximal average\R\datas\cod-rna.csv', 1, 1);
        E(abs(E) > 0.5) = 1;
        E(abs(E) == 0.5) = 1;
        E(abs(E) < 0.5) = 0;
        
        % define the problem
        [y_train, x_train] = libsvmread('D:\Proximal average\data\graph_guided_logistic_regression\cod-rna');
        x_train = full(x_train);
        x_train(:, 1) = featureScale(x_train(:, 1));
        x_train(:, 2) = featureScale(x_train(:, 2));
        x_train = x_train';
        y_train = y_train';
        lambda0 = 1e-4; lambda2 = 1e-4; lambda1 = 0;
        
    otherwise
            disp('ERROR!');
end
            
problem = graph_guided_logistic_regression(x_train, y_train, 0, 0, lambda1, lambda2, lambda0, E);

% solutions
f_opt = -inf;
w_opt = -inf;

% perform apa-svrg
% disp('=================== aga-svrg =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 20000);
% [~, infos_apa_svrg] = apa_svrg(problem, loc_options);

% perform admm-svrg
% disp('=================== admm-svrg ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 1, 'max_epoch', 10000);
% [w, infos_admm_svrg] = admm_svrg(problem, loc_options);

% perform apa-saga
% disp('====================== apa-saga =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 25000);
% [~, infos_apa_saga] = apa_saga(problem, loc_options);

% perform pa_asgd
% disp('====================== pa-asgd =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 30000);
% [~, infos_pa_asgd] = pa_asgd(problem, loc_options);

% perform svrg
% disp('====================== svrg =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.0001, 'max_epoch', 20000);
% [~, infos_svrg] = svrg(problem, loc_options);

% perform saga
disp('====================== saga =====================')
loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.0001, 'max_epoch', 30000);
[~, infos_saga] = saga(problem, loc_options);