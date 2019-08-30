%% german-numer
subplot(3, 2, 1)
n = 1000;

hold on
plot(infos1_apa_svrg.grad_calc_count(1: 10: 200)/n, infos1_apa_svrg.cost(1: 10: 200), 'k-', 'LineWidth', 2)
hold on
plot(infos1_apa_saga.grad_calc_count(1: 10: 400)/n, infos1_apa_saga.cost(1: 10: 400), 'm-', 'LineWidth', 2);
hold on
plot(infos1_pa_asgd.grad_calc_count(1: 300)/n, infos1_pa_asgd.cost(1: 300), '-', 'Color', [1 0.5 0], 'LineWidth', 2)
hold on
plot(infos1_admm_svrg.grad_calc_count(1: 100)/n, infos1_admm_svrg.cost(1: 100), 'g-', 'LineWidth', 2);
hold on
plot(infos1_svrg.grad_calc_count(1: 200)/n, infos1_svrg.cost(1: 200), 'k:', 'LineWidth', 2);
hold on
plot(infos1_saga.grad_calc_count(1: 300)/n, infos1_saga.cost(1: 300), 'b--', 'LineWidth', 2)

xlabel('Number of iterations');
ylabel('Objective value')
title('german-numer')
axis([0 20 0.47 0.65])
box on

%% a9a
subplot(3, 2, 2);
n = 32561;

hold on 
plot(infos2_apa_svrg.grad_calc_count(1: 100: 2000)/n, infos2_apa_svrg.cost(1: 100: 2000), 'k-', 'LineWidth', 2)
hold on
plot(infos2_apa_saga.grad_calc_count(1: 50: 8000)/n, infos2_apa_saga.cost(1: 50: 8000), 'm-', 'LineWidth', 2);
hold on
plot(infos2_pa_asgd.grad_calc_count(1: 10000)/n, infos2_pa_asgd.cost(1: 10000), '-', 'Color', [1 0.5 0], 'LineWidth', 2)
hold on
plot(infos2_admm_svrg.grad_calc_count(1: 2000)/n, infos2_admm_svrg.cost(1: 2000), 'g-', 'LineWidth', 2);
hold on
plot(infos2_svrg.grad_calc_count(1: 6000)/n, infos2_svrg.cost(1: 6000), 'k:', 'LineWidth', 2);
hold on
plot(infos2_saga.grad_calc_count(10: 8000)/n, infos2_saga.cost(10: 8000), 'b--', 'LineWidth', 2)

xlabel('Number of iterations');
ylabel('Objective value')
title('a9a')
axis([0 20 0.373 0.48]);
% axis([0 20 0.373 0.379]);
box on

%% w7a
subplot(3, 2, 3)
n = 24692;

hold on
plot(infos3_apa_svrg.grad_calc_count(1: 10: end)/n, infos3_apa_svrg.cost(1: 10: end), 'k-', 'LineWidth', 2)
hold on
plot(infos3_apa_saga.grad_calc_count(1: 10: end)/n, infos3_apa_saga.cost(1: 10: end), 'm-', 'LineWidth', 2);
hold on
plot(infos3_pa_asgd.grad_calc_count(1: 3000)/n, infos3_pa_asgd.cost(1: 3000), '-', 'Color', [1 0.5 0], 'LineWidth', 2)
hold on
plot(infos3_admm_svrg.grad_calc_count(1: end)/n, infos3_admm_svrg.cost(1: end), 'g-', 'LineWidth', 2);
hold on
plot(infos3_svrg.grad_calc_count(1: end)/n, infos3_svrg.cost(1: end), 'k:', 'LineWidth', 2);
hold on
plot(infos3_saga.grad_calc_count(1: end)/n, infos3_saga.cost(1: end), 'b--', 'LineWidth', 2)

xlabel('Number of iterations');
ylabel('Objective value')
title('w7a')
axis([0.0 12 0.15 0.45])
box on

%% cod-rna
subplot(3, 2, 4)
n = 59353;

hold on
line(1) = plot(infos4_apa_svrg.grad_calc_count(1: 10: end)/n, infos4_apa_svrg.cost(1: 10: end), 'k-', 'LineWidth', 2);
hold on
line(2) = plot(infos4_apa_saga.grad_calc_count(1: 10: end)/n, infos4_apa_saga.cost(1: 10: end), 'm-', 'LineWidth', 2);
hold on
line(3) = plot(infos4_pa_asgd.grad_calc_count(1: end)/n, infos4_pa_asgd.cost(1: end), '-', 'Color', [1 0.5 0], 'LineWidth', 2);
hold on
line(4) = plot(infos4_admm_svrg.grad_calc_count(1: end)/n, infos4_admm_svrg.cost(1: end), 'g-', 'LineWidth', 2);
hold on
line(5) = plot(infos4_svrg.grad_calc_count(1: end)/n, infos4_svrg.cost(1: end), 'k:', 'LineWidth', 2);
hold on
line(6) = plot(infos4_saga.grad_calc_count(1: end)/n, infos4_saga.cost(1: end), 'b--', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Objective value')
title('cod-rna')
axis([0.0 10 0.28 0.65])
box on

%% create overall legend
subplot(3, 2, [5 6])
axis off
l = legend(line(1:6), {'APA-SVRG', 'APA-SAGA', 'PA-ASGD', 'SVRG-ADMM', 'PA-SVRG', 'PA-SAGA'});
set(l,'Orientation','horizon')