function [W,b] = GetParams(d , K)
%GETPARAMS returns the initial parameters of the model (random guess))
W = 0.01*randn(K, d);
b = 0.01*randn(K, 1);

end

