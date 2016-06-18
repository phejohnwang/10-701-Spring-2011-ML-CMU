function [ prob ] = prob_y_x( k,w,x,i )
%P(Y=k|X=xi)
expSum = 0;
for l = 1:size(w,1)
    expSum = expSum + exp(w(l,:)*x(i,:)');
end
prob = exp(w(k,:)*x(i,:)') / expSum;
end

