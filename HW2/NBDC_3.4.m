% 10-701 Machine Learning, Spring 2011: Homework 2
% 3 Naive Bayes Document Classifier
% Question 3.4 by Pheno @ Gatech ECE
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
%alpha = 1 / length(vocabulary);
alpha = 0.75;   % change this to 'for' loop where log10(alpha) = linspace(-5,0,100);
% calculate theta
words = full(sum(count,1));  % word(k) - total # of words in category k news
for k = 1:length(label)
    for j = 1:length(vocabulary)
        theta(j,k) = (count(j,k) + alpha) / (words(k) + length(vocabulary) * alpha);
    end
end
%% Classifying Test Data
test_docID = testdata(:,1);
test_wordID = testdata(:,2);
test_count = testdata(:,3);

testP = zeros(length(testlabel),length(label)); % testP(m,k) - log10(P(Y=y_k|X)) for document m(docID)
% log10 used, so that testP will not go beyond matlab's real min / max
% initialize testP(m,k) with pi_k,
for k = 1:length(label)
    testP(:,k) = log10(ones(length(testlabel),1) * pi(k));
end
% calculate log10(P(Y) * P(X|Y))
thetaLog = log10(theta);   
for t = 1:length(test_docID);
    mm = test_docID(t);
    jj = test_wordID(t);
    for k = 1:length(label)
        testP(mm,k) = testP(mm,k) + test_count(t) * thetaLog(jj,k);
    end
end
% generate label for test data
[maxP,testlabel_predict] = max(testP,[],2);
%% Display Results
total_accuracy = sum(testlabel_predict == testlabel) / length(testlabel)

confusion = zeros(length(label));
for t = 1:length(testlabel);
    confusion(testlabel(t),testlabel_predict(t)) = confusion(testlabel(t),testlabel_predict(t)) + 1;
end
surf(confusion);
confusion
