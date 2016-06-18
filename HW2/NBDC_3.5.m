% 10-701 Machine Learning, Spring 2011: Homework 2
% 3 Naive Bayes Document Classifier
% Question 3.5~3.7 by Pheno @ Gatech ECE
%% Reading Data
clear all;
vocabulary = importdata('vocabulary.txt');
label = importdata('newsgrouplabels.txt');
traindata = importdata('train.data');
trainlabel = importdata('train.label');
testdata = importdata('test.data');
testlabel = importdata('test.label'); 
%% Training Data
%% pi Estimation
% P(Y=y_k) = pi_k
% MLE
pi_result = tabulate(trainlabel);
pi = pi_result(:,3)/100;
%% theta Estimation
% P(X=x_j|Y=y_k) = theta_j_k
% MAP
theta = zeros(length(vocabulary),length(label));
% count word frequency
count = sparse(length(vocabulary),length(label));   % count(j,k) - word frequecy of j(wordID) in category k(labelID) 
train_docID = traindata(:,1);
train_wordID = traindata(:,2);
train_count = traindata(:,3);
for t = 1:length(train_docID);
    jj = train_wordID(t);
    kk = trainlabel(train_docID(t));
    count(jj,kk) = count(jj,kk) + train_count(t);
end
%% change alpha
alpha = 1 / length(vocabulary);
% calculate theta
words = full(sum(count,1));  % word(k) - total # of words in category k news
for k = 1:length(label)
    for j = 1:length(vocabulary)
        theta(j,k) = (count(j,k) + alpha) / (words(k) + length(vocabulary) * alpha);
    end
end
%% Mutual Information
%I(word,Y) = H(Y) - H(Y|word) = H(word) - H(word|Y)
P_word = zeros(length(vocabulary),1);
H_word = zeros(length(vocabulary),1);
H_word_Y = zeros(length(vocabulary),1);
I_word_Y = zeros(length(vocabulary),1);
for j = 1:length(vocabulary)
    for k = 1:length(label)
        if theta(j,k)*(1-theta(j,k)) == 0
            tmp = 0;
        else
            tmp =  -theta(j,k)*log2(theta(j,k)) - (1-theta(j,k))*log2(1-theta(j,k));    %H(word|Y=y_k)
        end
        H_word_Y(j) = H_word_Y(j) + pi(k)*tmp;
    end
	P_word(j) = theta(j,:)*pi;	% P(word_j = true)
    if P_word(j)*(1-P_word(j)) == 0
        H_word(j) = 0;
    else
        H_word(j) = -P_word(j) * log2(P_word(j)) - (1-P_word(j))*log2(1-P_word(j));    
    end
    I_word_Y(j) = H_word(j) - H_word_Y(j);
end
%% Output Words with Highest Score - 100
[result,index] = sort(I_word_Y,1,'descend');
word_ = cell(100,1);
for i = 1:100
    word_(i) = vocabulary(index(i));
end
word_
