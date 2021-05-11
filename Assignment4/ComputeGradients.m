function gradients = ComputeGradients(RNN, X, Y, a, h, p, n,m)
    W = RNN.W;
    V = RNN.V;
    g_h = zeros(n, m);
    g_a = zeros(n, m);  
    
    g = -(Y-p)';
    gradients.c = (sum(g))';
    gradients.V = g'*h(:, 2:end)';
    
    g_h(n,:) = g(n,:)*V;
    g_a(n, :) = g_h(n, :)*diag(1 - (tanh(a(:, n))).^2); 
    
    for t = (n-1):-1:1
        g_h(t, :) = g(t, :)*V + g_a(t+1, :)*W;
        g_a(t, :) = g_h(t, :)*diag(1 - (tanh(a(:, t))).^2);
    end
    
    gradients.b = (sum(g_a))';
    gradients.W = g_a'*h(:,1:end-1)';
    gradients.U = g_a'*X';
    
end

