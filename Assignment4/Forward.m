function [loss, a, h, o, p] = Forward(RNN, X, Y, h0, n, K, m)

o = zeros(K, n);
p = zeros(K, n);
h = zeros(m, n);
a = zeros(m, n);
ht = h0;
loss = 0;

for t = 1:n
    at = RNN.W*ht +RNN.U*X(:,t) + RNN.b;
    a(:,t) = at;
    ht = tanh(at);
    h(:,t) = ht;
    o(:,t) = RNN.V*ht + RNN.c;
    p(:,t) = exp(o(:,t))/sum(exp(o(:,t)));
    
    loss = loss - log(Y(:,t)'*p(:,t));
end
h = [h0, h];

end

