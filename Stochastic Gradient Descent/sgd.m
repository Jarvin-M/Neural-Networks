function [w1,w2,E_cost,test_error] = sgd(xi,tau, P,Q,eta,tmax)
% Input
% eta - learning rate
% P - number of training sets
% Q -number of test sets
% xi - input vectors 50 x 5000 vector
% tau - corresponding labels
% eta - learning rate
% tmax - number of learning steps

    
    
    % w1 and w2 - N dim vectors of adpative input to hidden networks
    %Initialize the weights as independent random vectors with |w1|^2 = 1 and
    %|w2|^2 = 1.
    N = length(xi(:,1));
    w1 = rand(N,1);
    w1 = w1/sum(w1);
    w2 = rand(N,1);
    w2 = w2/sum(w2);

    E_cost = zeros(tmax,1); % stores the cost function E(t)
    test_error = zeros (tmax,1); % E_test(t)
    % Stochastic gradient descent procedure
    for i = 1: tmax
  
        for example = 1:P
            % In each learning step, select one of the P example randomly
            index = randi(P); % equal probability
            %gradient with respect to random input
            xi_v = xi(:,index);
            tau_v = tau(index);
            t1 = dot(w1,xi_v);
            t2 = dot(w2,xi_v);
            sigma = tanh(t1) +tanh(t2);
            
            %Ev = (sigma - tau_v)^2 /2;
            % from gradient-example.pdf
            
            delta1 = (sigma - tau_v)* (1-tanh(t1)^2) * xi_v; %gradient with w.r.t w1
            delta2 = (sigma - tau_v)* (1-tanh(t2)^2) * xi_v; %gradient  with w.r.t w2

            w1 = w1 - (eta*delta1); % update of
            w2 = w2 - (eta*delta2); 
        end
        
        %Compute E and Etest after P single randomized steps, not after each individual update
        [E,E_test] = cost(w1,w2,xi,tau,P,Q);
        
        E_cost(i) = E;
        test_error(i) = E_test;

    end
end

