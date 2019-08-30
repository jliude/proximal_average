% an experiment for overlapping group lasso

clc;
clear all;
close all;

% generate the dataset
K = 50; % number of groups
d = 90*K + 10; % dimensions
n = d; % 
% 
% [X, y, G] = generate_data1(K);
load('D:\Proximal average\data\overlapping_group_lasso\K_50\data.mat');

lambda0 = K/(5 * n);
lambda1 = 0;
lambda2 = 0.0;

% define problem
problem = overlapping_group_lasso_regression(X, y, 0, 0, lambda1, lambda2, lambda0, G);

% solutions
f_opt = -inf;
w_opt = -inf;

% % perform ppag
% disp('=================== ppag ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.001, 'max_epoch', 30000);
% [~, infos_ppag] = ppag3(problem, loc_options);

% % perform admm-svrg
disp('=================== admm-svrg ====================')
loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.00025, 'max_epoch', 600);
[w, infos_admm_svrg] = admm_svrg(problem, loc_options);

% perform apa-svrg
% disp('=================== aga-svrg =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 30);
% [~, infos_apa_svrg] = apa_svrg(problem, loc_options);

% perform svrg
% disp('====================== svrg =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.00001, 'max_epoch', 80);
% [~, infos_svrg] = svrg(problem, loc_options);

% perform saga
% disp('====================== saga =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.00001, 'max_epoch', 400);
% [~, infos_saga] = saga(problem, loc_options);

% perform apa-saga
% disp('====================== apa-saga =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'stepsize', 0.0001, 'max_epoch', 30);
% [~, infos_apa_saga] = apa_saga(problem, loc_options);

% perform svrg
% disp('====================== pa-asgd =====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 400);
% [~, infos_pa_asgd] = pa_asgd(problem, loc_options);