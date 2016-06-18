% 10-701 Machine Learning, Spring 2011: Homework 3
% 5 Handwritten Digit Recognition
% Question 2.5 a) by Pheno @ Gatech ECE
%% Read Data
clear all;
load('usps_digital.mat'); 
%% Initialize w
w = zeros(K,d);
% Calculate L(w_k)
L0 = log_likelihood(w,K,d,tr_X,tr_y);
% Calculate training accuracy
Tr_A0 = CalAccuracy(w,tr_X,tr_y);
% Calculate testing accuracy
Te_A0 = CalAccuracy(w,te_X,te_y);
%% Gradient Ascent Algorithm - M times
stepsize = 0.0001;
M = 500;
Tr_A = zeros(M,1);
Te_A = zeros(M,1);
L = zeros(M,1);
for m = 1:M
    % Update w
    w = gradient_ascent( w, K, d, tr_X, tr_y, stepsize );
    % Calculate L(w_k)
    L(m) = log_likelihood(w,K,d,tr_X,tr_y);
    % Training accuracy
    Tr_A(m) = CalAccuracy(w,tr_X,tr_y);
    % Testing accuracy
    Te_A(m) = CalAccuracy(w,te_X,te_y);
    % Display results
    if (m<10 || mod(m,100)==0)
        fprintf('Iter=%d, Obj=%f, tr_acc=%f, te_acc=%f\n', m, L(m), Tr_A(m), Te_A(m));
	end
end
%% Plot
x = 0:M;
L_likelihood = [L0 L'];
Tr_Acuracy = [Tr_A0 Tr_A'];
Te_Acuracy = [Te_A0 Te_A'];
figure;
plot(x,L_likelihood);
figure;
plot(x,Tr_Acuracy,x,Te_Acuracy);
legend('Training','Testing');