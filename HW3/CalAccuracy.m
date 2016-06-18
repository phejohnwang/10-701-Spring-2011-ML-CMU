function [ accuracy ] = CalAccuracy( w,x,y )
% Calculate the accuracy
result = exp(w*x');
[max1,predict] = max(result);
correct = sum(predict == y');
accuracy = correct/length(y);
end

