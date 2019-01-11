data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

eta = 0.05; % learning rate
P =100; % number of training sets
Q= 120; % number of test sets. Shouls be 100 or larger
tmax =50; 
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
xlabel("learning steps(t)");
ylabel("Cost function E(t)");
saveas(gcf, 'costvst.png');
% 
% 
% Plot and compare the evolution of E and Etest with the training time t
figure()
plot((1:tmax),mean(all_e,2),"linewidth",1.2);
hold on;
plot((1:tmax),mean(all_test,2),"linewidth",1.2);
% title("Evolutions of averages of E(t) and Etest(t) with learning steps (t) over multiple runs")
xlabel("learning steps(t)");
ylabel("value");
legend("Cost Function E(t)", "Test/Generalization error Etest(t)");
saveas(gcf, 'costvsgeneralisation.png');

%display the obtained, final weight vectors, for instance as bar graphs.
figure();
bar(mean(all_w1,2));
hold on;
bar(mean(all_w2,2));
ylabel("Weight value");
xlabel("Element index of weight vector");
% title("Average final weights for 40 random initialisations");
legend("Final weights w1", "Final weights w2");
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
    plot((1:tmax),E_cost,"linewidth",1.2,'DisplayName',"P = Q = "+p_each);
    hold on;
    set(0, 'CurrentFigure', f2)
    plot((1:tmax),testError,"linewidth",1.2,'DisplayName',"P = Q = "+p_each);
    hold on;
    
end
% title("Cost function E(t) vs t for P =[20, 50, 200, 500, 1000, 2000]")
set(0, 'CurrentFigure', f1);
xlabel("learning steps(t)");
ylabel("Cost function E(t)");
legend();
saveas(gcf, 'differentP.png')

set(0, 'CurrentFigure', f2)
xlabel("learning steps(t)");
ylabel("Generalisation error E(t)");
legend();
saveas(gcf, 'differentPerror.png')
% influence of the learning rate ?. Investigating convergence for a larger
% training set
eta_list = [0.05 0.025 0.001 0.0001]; % learning rate
PP =2000; % number of training sets
QQ= 2200;

figure();
for et = eta_list % wo runs with random initialisations
    all_cost =[];
    for runs =1: 50
    [~,~,E_cost,~] = sgd(xi,tau, PP,QQ,et,30);
    all_cost(:,runs) = E_cost;
    end
    plot((1:30),mean(all_cost,2),"linewidth",1.2,'DisplayName',"\eta = "+et );
    hold on;
end
xlabel("learning steps(t)");
ylabel("Cost function E(t)");
legend();
saveas(gcf, 'differentEta.png')