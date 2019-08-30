plot(infos_svrg.grad_calc_count(50: 500), infos_svrg.cost(50: 500), 'k:', 'LineWidth', 1.5)
hold on
plot(infos_apa_svrg.grad_calc_count(2: 500), infos_apa_svrg.cost(2: 500), 'r-', 'LineWidth', 1.5)
hold on
plot(infos_admm_svrg.grad_calc_count(2: 500), infos_admm_svrg.cost(2: 500), 'b', 'LineWidth', 1.5)
hold on
plot(infos_apa_saga.grad_calc_count(2: 500), infos_apa_saga.cost(2: 500), 'g', 'LineWidth', 1.5)
hold on
plot(infos_saga.grad_calc_count(5: 500), infos_saga.cost(5: 500), 'c', 'LineWidth', 1.5)
hold on
plot(infos_pa_asgd.grad_calc_count(2: 500), infos_pa_asgd.cost(2: 500), 'y', 'LineWidth', 1.5)
