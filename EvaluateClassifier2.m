function P = EvaluateClassifier2(X, W, b)
    %EVALUATECLASSIFIER that evaluates the network function, i.e. equations
    %(1, 2), on multiple images and returns the results.

    %Each column of X corresponds to an image and it has size dxn
    %W and b are the parameters of the network
    %Each column of P contains the probability for each label for the image 
    %in the corresponding column of X. P has size Kxn

    s = W*X + b;
    P = softmax(s);

end

