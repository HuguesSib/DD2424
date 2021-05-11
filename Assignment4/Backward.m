function [RNN,M] = Backward(RNN, X, Y, a, h, p,eta, n, m,M)
    gradients = ComputeGradients(RNN, X, Y, a, h, p, n, m);
    eps = 1e-8;

    for f=fieldnames(gradients)'
        gradients.(f{1}) = max(min(gradients.(f{1}), 5), -5);
    end
    
    for f = fieldnames(RNN)'
        % AdaGrad
        M.(f{1}) = M.(f{1}) + gradients.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta*(gradients.(f{1})./(M.(f{1}) + eps).^(0.5));
        %RNN.(f{1}) = RNN.(f{1}) - eta*(gradients.(f{1}));
    end
end

