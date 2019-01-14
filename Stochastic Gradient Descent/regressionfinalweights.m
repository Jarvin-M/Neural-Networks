clearvars
close all

rng default

data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

P = 1000; % optimal parameters
Q = 1000;
eta = 0.001;
tmax = 50;

for run = 1:100 % wo runs with random initialisations
    [w1,w2,E_cost,test_error] = sgd(xi,tau, P,Q,eta,tmax);
    all_e(:,run) = E_cost;
    all_test(:,run) = test_error;
    all_w1(:,run) = w1;
    all_w2(:,run) = w2;
end


%display the obtained, final weight vectors, for instance as bar graphs.
figure();
bar(1:50, [mean(all_w1,2) mean(all_w2,2)], 'BarWidth', 1);
ylabel("Weight value", 'Interpreter', 'latex');
xlabel("Component index", 'Interpreter', 'latex');
% title("Average final weights for 40 random initialisations");
legend("Final weights $\mathbf w_1$", "Final weights $\mathbf w_2$", 'Interpreter', 'latex');
saveas(gcf, 'finalfinalweights.png');