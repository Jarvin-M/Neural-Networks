function s = sigma(w1, w2, xi)
    s = tanh(dot(w1, xi)) + tanh(dot(w2, xi));
end