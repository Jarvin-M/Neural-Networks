clearvars
close all

rng default

data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

P = 1000; % optimal parameters
Q = 1000;
eta = 0.001;

% adjustable hidden-to-output
colors = linspecer(5);
coliter = 1

figure();
hold on;
all_cost =[];
all_t =[];
for runs =1: 50
    [~,~,E_cost,test_Error] = sgd(xi,tau, P,Q,0.001,30);
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2 );
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

all_cost =[];
all_t =[];
v = []
for runs =1: 50
    [~,~,vtemp,E_cost,test_Error] = sgdadj(xi,tau, P,Q,0.001, 0.001,30);
    v(:, runs) = vtemp;
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2);
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

all_cost =[];
all_t =[];
v = []
for runs =1: 50
    [~,~,vtemp,E_cost,test_Error] = sgdadj(xi,tau, P,Q,0.001, 0.0001,30);
    v(:, runs) = vtemp;
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2);
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

all_cost =[];
all_t =[];
v = []
for runs =1: 50
    [~,~,vtemp,E_cost,test_Error] = sgdadj(xi,tau, P,Q,0.001, 0.00001,30);
    v(:, runs) = vtemp;
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2);
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1

all_cost =[];
all_t =[];
v = []
for runs =1: 50
    [~,~,vtemp,E_cost,test_Error] = sgdadj(xi,tau, P,Q,0.001, 0.000001,30);
    v(:, runs) = vtemp;
    all_cost(:,runs) = E_cost;
    all_t(:,runs) = test_Error;
end
plot((1:30),mean(all_cost,2),'color',colors(coliter, :),"linewidth",1.2);
plot((1:30),mean(all_t,2),'--','color',colors(coliter, :),"linewidth",1.2 );
coliter = coliter + 1


mean(v, 2)

xlabel("Learning step $t$", 'Interpreter', 'latex');
ylabel("Average cost function $E(t)$", 'Interpreter', 'latex');
legend("$v_1 = v_2 = 1$", "$v_1 = v_2 = 1$", "adjustable $v_1, v_2$, $\eta_v = 0.001$", ...
    "adjustable $v_1, v_2$, $\eta_v = 0.001$", "adjustable $v_1, v_2$, $\eta_v = 10^{-4}$", ...
    "adjustable $v_1, v_2$, $\eta_v = 10^{-4}$", "adjustable $v_1, v_2$, $\eta_v = 10^{-5}$", ...
    "adjustable $v_1, v_2$, $\eta_v = 10^{-5}$", "adjustable $v_1, v_2$, $\eta_v = 10^{-6}$", ...
    "adjustable $v_1, v_2$, $\eta_v = 10^{-6}$", 'Interpreter', 'latex');
saveas(gcf, 'adjustable.png')