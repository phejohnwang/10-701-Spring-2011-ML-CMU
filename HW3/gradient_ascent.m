function [ w_new ] = gradient_ascent( w, K, d, x, y, stepsize )
% Gradient Ascent Algorithm - One iteration
% Calculate gradient of L
Gradient = zeros(K,d);
for k=1:K-1 % w(K) remains constant (= 0)
    for i = 1:length(y)    
        tmpValue = (y(i)==k) - prob_y_x(k,w,x,i); %P(Y=k|X=xi)
        Gradient(k,:) = Gradient(k,:) +  x(i,:) * tmpValue;
    end
end
% Update w
w_new = w + stepsize * Gradient;
end

