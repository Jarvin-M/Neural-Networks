clear all
close all
rng default                             % set seed

N = 20;                                 % dimension of feature vectors
nmax = 100;                             % maximum number of epochs
nD = 50;                                % number of dichotomies

alpha_min = 0.75;                       % specify values of alpha
alpha_max = 3;
d_alpha = .25;

alphaList = alpha_min:d_alpha:alpha_max;

Qls = [];


for alpha = alphaList
    P = round(alpha * N);                   % number of feature vectors

    nsucc = 0;                              % number of successful runs

    for dichotomy = 1:nD                    % repeat for nD dichotomies

        data = randn(N, P);                     % draw ND-data from N(0, I)
        labels = 1 - 2 * randi([0, 1], 1, P);   % generate labels

        w = zeros(N, 1);                        % initial weights
        solution = 0;                           % reset

        for epoch = 1:nmax
            for example = 1:P
                E = w' * data .* labels;        % determine local potentials

                solution = prod(E > 0);         % 1 iff [each component of E] > 1 
                if (solution)                   % stop if a solution is found
                    break
                end

                if (E(example) <= 0)            % update w
                    w = w + data(:, example) * labels(example);
                end
            end

            if (solution)
                break
            end
        end

        nsucc = nsucc + solution;
    end

    Qls = [Qls, nsucc / nD];            % convert to fraction and add to Qls
end

figure
plot(alphaList, Qls, '-')
xlabel("alpha")
ylabel("fraction of linearly separable dichotomies")