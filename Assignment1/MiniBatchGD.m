function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, eta)
P = EvaluateClassifier(X, W, b);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);

%Update parameters
Wstar = W-eta*grad_W;
bstar = b-eta*grad_b;
end

