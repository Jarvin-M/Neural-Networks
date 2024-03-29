clearvars
close all

data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

eta = 0.05; % learning rate
P =100; % number of training sets
Q= 100; % number of test sets. Shouls be 100 or larger
tmax =500; 
%consider the first P=100
% average out multiple runs
all_e =[]; % store all cost functions over multiple runs
all_test =[];
all_w1 = [];
all_w2 =[];

for run = 1:100 % wo runs with random initialisations
    [w1,w2,E_cost,test_error] = sgd(xi,tau, P,Q,eta,tmax);
    all_e(:,run) = E_cost;
    all_test(:,run) = test_error;
    all_w1(:,run) = w1;
    all_w2(:,run) = w2;
end



%plot E vs the time t
figure();
plot((1:tmax),mean(all_e,2),"linewidth",1.2); % mean(all_e,2) - averages out the cost functions for each t over multiple runs
% title("Average Cost function E(t) vs t for 100 random runs")
xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Cost function $E(t)$", 'Interpreter', 'latex');
saveas(gcf, 'costvst.png');
% 
% 
% Plot and compare the evolution of E and Etest with the training time t
figure()
plot((1:tmax),mean(all_e,2),"linewidth",1.2);
hold on;
plot((1:tmax),mean(all_test,2),"linewidth",1.2);
% title("Evolutions of averages of E(t) and Etest(t) with learning steps (t) over multiple runs")
xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Value", 'Interpreter', 'latex');
legend("Cost function $E(t)$", "Generalization error $E_{test}(t)$", 'Interpreter', 'latex');
saveas(gcf, 'costvsgeneralisation.png');

%display the obtained, final weight vectors, for instance as bar graphs.
figure();
bar(1:50, [mean(all_w1,2) mean(all_w2,2)], 'BarWidth', 1);
ylabel("Weight value", 'Interpreter', 'latex');
xlabel("Component index", 'Interpreter', 'latex');
% title("Average final weights for 40 random initialisations");
legend("Final weights $\mathbf w_1$", "Final weights $\mathbf w_2$", 'Interpreter', 'latex');
saveas(gcf, 'finalweights.png');

% Different values of P and Q
various_P = [20 50 200 500 1000 2000]; % considering P=Q

f1 = figure;
f2 = figure;
for p_each = various_P
    Q = p_each;
    [~,~,E_cost,testError] = sgd(xi,tau, p_each,Q,eta,tmax);
    %plot E vs the time t
    set(0, 'CurrentFigure', f1)
    plot((1:tmax),E_cost,"linewidth",1.2);
    hold on;
    set(0, 'CurrentFigure', f2)
    plot((1:tmax),testError,"linewidth",1.2);
    hold on;
    
end
% title("Cost function E(t) vs t for P =[20, 50, 200, 500, 1000, 2000]")
set(0, 'CurrentFigure', f1);
xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Cost function $E(t)$", 'Interpreter', 'latex');
legend("$P = Q = 20$", "$P = Q = 50$", "$P = Q = 200$", "$P = Q = 500$", ...
    "$P = Q = 1000$", "$P = Q = 2000$", 'Interpreter', 'latex');
saveas(gcf, 'differentP.png')

set(0, 'CurrentFigure', f2)
xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Generalization error $E_{test}(t)$", 'Interpreter', 'latex');
legend("$P = Q = 20$", "$P = Q = 50$", "$P = Q = 200$", "$P = Q = 500$", ...
    "$P = Q = 1000$", "$P = Q = 2000$", 'Interpreter', 'latex');
saveas(gcf, 'differentPerror.png')
% influence of the learning rate ?. Investigating convergence for a larger
% training set
eta_list = [0.05 0.025 0.001 0.0001]; % learning rate
PP =1000; % number of training sets
QQ= 1000;

colors = linspecer(length(eta_list));
coliter = 1
figure();
hold on;
for et = eta_list % wo runs with random initialisations
    all_cost =[];
    for runs =1: 50
        [~,~,E_cost,test_Error] = sgd(xi,tau, PP,QQ,et,30);
        all_cost(:,runs) = E_cost;
        all_t(:,runs) = test_Error;
    end
    plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2,'DisplayName',"\eta = "+et );
    plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2,'DisplayName',"\eta = "+et );
    coliter = coliter + 1
end
xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Average cost function $E(t)$", 'Interpreter', 'latex');
legend();
saveas(gcf, 'differentEta.png')





