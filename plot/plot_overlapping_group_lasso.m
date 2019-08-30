% K = 5
subplot(3, 2, 1);
K = 5;
n = 90*K + 10;

hold on
plot(infos1_apa_svrg.grad_calc_count(1: 500)/n, infos1_apa_saga.cost(1: 500), 'k-', 'LineWidth', 2);
hold on
plot(infos1_apa_saga.grad_calc_count(1: 500)/n, infos1_apa_saga.cost(1: 500), 'm-', 'LineWidth', 2);
hold on
plot(infos1_pa_asgd.grad_calc_count(1: 500)/n, infos1_pa_asgd.cost(1: 500), '-', 'Color', [1 0.5 0], 'LineWidth', 2);
hold on
plot(infos1_admm_svrg.grad_calc_count(1: 500)/n, infos1_admm_svrg.cost(1: 500), 'g-', 'LineWidth', 1.5);
hold on
plot(infos1_svrg.grad_calc_count(1: 500)/n, infos1_svrg.cost(1: 500), 'k:', 'LineWidth', 2);
hold on
plot(infos1_saga.grad_calc_count(1: 500)/n, infos1_saga.cost(1: 500), 'b--', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Objective value')
title('K = 5')
axis([0 100 0 8]);
box on


% K = 10
subplot(3, 2, 2);
K = 10;
n = 90*K + 10;

hold on
plot(infos2_apa_svrg.grad_calc_count(1: 500)/n, infos2_apa_saga.cost(1: 500), 'k-', 'LineWidth', 2);
hold on
plot(infos2_apa_saga.grad_calc_count(1: 500)/n, infos2_apa_saga.cost(1: 500), 'm-', 'LineWidth', 2);
hold on
plot(infos2_pa_asgd.grad_calc_count(1: 500)/n, infos2_pa_asgd.cost(1: 500), '-', 'Color', [1 0.5 0], 'LineWidth', 2);
hold on
plot(infos2_admm_svrg.grad_calc_count(1: 500)/n, infos2_admm_svrg.cost(1: 500), 'g-', 'LineWidth', 1.5);
hold on
plot(infos2_svrg.grad_calc_count(1: 500)/n, infos2_svrg.cost(1: 500), 'k:', 'LineWidth', 2);
hold on
plot(infos2_saga.grad_calc_count(1: 500)/n, infos2_saga.cost(1: 500), 'b--', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Objective value')
title('K = 10')
axis([0 55 0 8])
box on


% K = 20
subplot(3, 2, 3);
K = 20;
n = 90*K + 10;

hold on
plot(infos3_apa_svrg.grad_calc_count(1: 1000)/n, infos3_apa_saga.cost(1: 1000), 'k-', 'LineWidth', 2);
hold on
plot(infos3_apa_saga.grad_calc_count(1: 1000)/n, infos3_apa_saga.cost(1: 1000), 'm-', 'LineWidth', 2);
hold on
plot(infos3_pa_asgd.grad_calc_count(1: 1200)/n, infos3_pa_asgd.cost(1: 1200), '-', 'Color', [1 0.5 0], 'LineWidth', 2);
hold on
plot(infos3_admm_svrg.grad_calc_count(1: 1000)/n, infos3_admm_svrg.cost(1: 1000), 'g-', 'LineWidth', 1.5);
hold on
plot(infos3_svrg.grad_calc_count(1: 1000)/n, infos3_svrg.cost(1: 1000), 'k:', 'LineWidth', 2);
hold on
plot(infos3_saga_000001.grad_calc_count(1: 1000)/n, infos3_saga_000001.cost(1: 1000), 'b--', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Objective value')
title('K = 20')
axis([0 50 0 8])
box on

% K = 50
subplot(3, 2, 4)
K = 50;
n = 90*K + 10;

hold on
line(1) = plot(infos4_apa_svrg.grad_calc_count(1: 1500)/n, infos4_apa_saga.cost(1: 1500), 'k-', 'LineWidth', 2);
hold on
line(2) = plot(infos4_apa_saga.grad_calc_count(1: 2000)/n, infos4_apa_saga.cost(1: 2000), 'm-', 'LineWidth', 2);
hold on
line(3) = plot(infos4_pa_asgd.grad_calc_count(1: 2000)/n, infos4_pa_asgd.cost(1: 2000), '-', 'Color', [1 0.5 0], 'LineWidth', 2);
hold on
line(4) = plot(infos4_admm_svrg.grad_calc_count(1: 632)/n, infos4_admm_svrg.cost(1: 632), 'g-', 'LineWidth', 1.5);
hold on
line(5) = plot(infos4_svrg_000001.grad_calc_count(1: 1500)/n, infos4_svrg_000001.cost(1: 1500), 'k:', 'LineWidth', 2);
hold on
line(6) = plot(infos4_saga_000001.grad_calc_count(1: 2000)/n, infos4_saga_000001.cost(1: 2000), 'b--', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Objective value')
title('K = 50')
axis([0 45 0 8])
box on

% create overall legend
subplot(3, 2, [5 6])
axis off
l = legend(line(1:6), {'APA-SVRG', 'APA-SAGA', 'PA-ASGD', 'SVRG-ADMM', 'SVRG', 'SAGA'});
set(l,'Orientation','horizon')