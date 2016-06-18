function [ L ] = log_likelihood( w,K,d,x,y )
% Calculate log_likelihood from w, x and y
L = 0;
for i=1:length(y)
    sumI = w(y(i),:) * x(i,:)';    
    tmp = 0;    
    for l=1:K
        tmp = tmp + exp(w(l,:) * x(i,:)');
    end
    sumI = sumI - log(tmp);
    L = L + sumI; 
end
end

