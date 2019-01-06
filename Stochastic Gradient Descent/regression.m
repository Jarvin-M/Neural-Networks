data = load("data3.mat");
xi = data.xi; % 50 x 5000 array
tau = data.tau; % 5000 target values

eta = 0.05; % learning rate
P =100; % number of training sets
Q= 120; % number of test sets. Shouls be 100 or larger
tmax =50; 
%consider the first P=100

[w1,w2,E_cost,test_error] = sgd(xi,tau, P,Q,eta,tmax);

%plot E vs the time t
figure();
plot((1:tmax),E_cost);
title("Cost function E(t) vs t")
xlabel("learning steps(t)");
ylabel("Cost function E(t)");


% Plot and compare the evolution of E and Etest with the training time t
figure()
plot((1:tmax),E_cost);
hold on;
plot((1:tmax),test_error);
title("Evolutions of E(t) and Etest(t) with learning steps (t)")
xlabel("learning steps(t)");
ylabel("value");
legend("Cost Function E(t)", "Test/Generalization error Etest(t)");

%display the obtained, final weight vectors, for instance as bar graphs.

