function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
%COMPUTEGRADIENTS evaluates, for a mini-batch, the gradients of the cost 
%function w.r.t. W and b, that is equations (10, 11). 
%X has size dxn
%Y has size Kxn
%P contains the probability for each label for the image in the
%corresponding column, has size Kxn

%grad_W is the gradient matrix of the cost J relative to W, has size 
%Kxd
%grad_b is the gradient vector of the cost J relative to b, has size 
%Kx1
    grad_W = zeros(size(W));
    grad_b = zeros(size(W, 1), 1);

    for i = 1 : size(X, 2)
        P_i = P(:, i);
        Y_i = Y(:, i);
        X_i = X(:, i);        
        
        g = -(Y_i-P_i)';
        grad_b = grad_b + g';
        grad_W = grad_W + g'*X_i';
 
    end
    % divide grad by the number of entries in D
    grad_b = grad_b/size(X, 2);
    grad_W = grad_W/size(X, 2) + 2*lambda*W ;
 end

