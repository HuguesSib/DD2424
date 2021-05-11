function acc = ComputeAccuracy1(X, y, W, b)
    %COMPUTEACCURACY computes the accuracy of the network’s predictions given 
    %by equation (4) on a set of data.
    P = EvaluateClassifier2(X,W,b);
    [~, pred] = max(P);
    acc = sum(pred == y)/length(P);
end

