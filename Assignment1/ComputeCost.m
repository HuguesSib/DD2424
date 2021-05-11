function J = ComputeCost(X,Y,W,b, lambda)
    %COMPUTECOST  computes the cost function given by equation
    %(5) for a set of images
    P = EvaluateClassifier(X,W,b);

    J= -mean(Y.*P)/size(X,2) + lambda*sumsqr(W);
end

