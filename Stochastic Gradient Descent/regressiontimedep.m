clearvars
close all

data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

PP =1000; % number of training sets
QQ= 1000;


% time dependent learning rates
etafunc_list = [0]
colors = linspecer(2);
coliter = 1

figure();
hold on;
all_cost =[];
all_t =[];
for runs =1: 50
    [~,~,E_cost,test_Error] = sgd(xi,tau, PP,QQ,0.001,30);
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2 );
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

all_cost =[];
all_t =[];
for runs =1: 50
    [~,~,E_cost,test_Error] = sgdvar(xi,tau, PP,QQ,30/1000, 0,30);
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2);
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Average cost function $E(t)$", 'Interpreter', 'latex');
legend("$\eta(t) = .001$", "$\eta(t) = .001$", "$\eta(t) = 30/(1000 t)$", ...
    "$\eta(t) = 30/(1000 t)$", 'Interpreter', 'latex');
saveas(gcf, 'differentEtaFunc.png')
xlim([0, 30])
xticks(0:2:30)
saveas(gcf, 'differentEtaFuncZoomed.png')