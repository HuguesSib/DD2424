function p = softmax(s)
%Compute the softmax of S, return a probability vector
p = exp(s)./sum(exp(s));
end

