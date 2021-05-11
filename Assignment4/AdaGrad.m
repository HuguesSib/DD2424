function [RNN, loss_s, min_RNN, min_h, min_iter, min_loss, M,iter] = AdaGrad(RNN, data,  n, K, m, eta, iter, ind_to_char, smooth_loss, min_loss, M)
%AdaGrad algorithm
e =1;
len = 200;
loss_s = [];

while e <= length(data) -n -1
    Xbatch = data(:, e: e+n-1);
    Ybatch = data(:, e +1 :e+n);
    if e ==1
        hprev = zeros(m,1);
    else
        hprev = h(:,end); 
    end
    
    [loss, a, h, ~, p] = Forward(RNN, Xbatch, Ybatch, hprev, n, K, m);
    [RNN, M] = Backward(RNN, Xbatch, Ybatch, a,h, p, eta, n, m, M);
    
    if iter == 1 && e ==1 
        %initialize first ite
        smooth_loss = loss;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    
    if smooth_loss < min_loss
        min_RNN = RNN;
        min_h = hprev;
        min_iter = iter;
        min_loss = smooth_loss;
    end
    
    loss_s = [loss_s, smooth_loss];
    
    if iter == 1 || mod(iter, 10000) ==0
        y = synthesize(RNN, hprev, data(:,1), len, K);
        c = [];
        for i = 1:len
            c = [c ind_to_char(y(i))];
        end
        disp ("---------------------------------");
        disp (["iter = " iter ", smooth_loss = " smooth_loss]);
        disp(c);
    end
    iter = iter +1;
    e = e + n;
end
end

