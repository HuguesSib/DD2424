function y = synthesize(RNN,h0, x0, n ,K)
%The function will synthesize a sequence of characters using the current
%parameter values in RNN.
%h0 : hidden state at time 0
%x0 : the first(dummy) input to the RNN
%n : denotes the length of the sequence e want to generate
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
h = h0;
x = x0;
y = zeros(1,n);
for t= 1:n
    a = W*h + U*x + b;
    h = tanh(a);
    o = V*h + c;
    p = exp(o)/sum(exp(o)); 
    
    %Randomly select a character based on the output probability scores p
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    
    %generate the next output
    N = length(ii);
    x = zeros(K,N);
    for i = 1:N 
        x(ii(i),i) = 1; %One hot encoding of next input
    end
    y(t) = ii;
end
end

