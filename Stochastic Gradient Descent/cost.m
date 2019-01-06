function [E,E_test] = cost(w1,w2,xi,tau,P,Q)
%Compute E and Etest after P single randomized steps, not after each individual update. It is
%recommended to define a (matlab) function which calculates E and Etest with w1, w2 and the
%corresponding data set (inputs and labels) as arguments.

% input
% P - number of training sets
% Q - number of test sets - Q -100 or larger
% w1 and w2 - weight vectors
% xi - input vector 50 X 5000
% tau - correpsonding continuos labels

% E - costfunction
%E_test - test/generalization error

% cost function
ee_sum =0;
for mu= 1:P
    xi_mu = xi(:,mu);
    tau_mu = tau(mu);% label of correspondin xi_mu
    t1 = dot(w1,xi_mu);
    t2 = dot(w2,xi_mu);
    sigma = tanh(t1) +tanh(t2);
    ee = (sigma - tau_mu)^2;
    ee_sum = ee_sum + ee;
end

E = ee_sum /(2*P);

% test/generalization error

ee_test =0;
for mu_test= (P+1): P+Q
    xi_mutest = xi(:,mu_test);
    tau_mutest= tau(mu_test);% label of correspondin xi_mu
    t11 = dot(w1,xi_mutest);
    t22 = dot(w2,xi_mutest);
    sigma_test = tanh(t11) +tanh(t22);
    ee_t = (sigma_test - tau_mutest)^2;
    
    ee_test = ee_test+ ee_t;
end

E_test = ee_test / (2*Q);
end

