function loss = ComputeLoss(X, Y , RNN, h)
    %COMPUTELOSS 
    n = size(X,2);
    loss = 0;

    for i = 1:n
        a = RNN.W*h + RNN.U*X + RNN.b;
        h = tanh(a);
        o = RNN.V*h + RNN.c;
        p = exp(o)/sum(exp(o)); 
        
        loss = loss - log(Y(:,i)'*p);
    end
end

