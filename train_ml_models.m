clear; clc; close all;

load('training_dataset.mat'); 

cv = cvpartition(all_labels, 'HoldOut', 0.30);
Xtrain = all_features(training(cv), :);
Ytrain = all_labels(training(cv), :);
Xtest  = all_features(test(cv), :);
Ytest  = all_labels(test(cv), :);

%% ===================== Train SVM =====================
fprintf('Training SVM...\n');
svm_model = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'Standardize', true);
svm_pred = predict(svm_model, Xtest);
svm_accuracy = mean(svm_pred == Ytest);
fprintf('SVM Accuracy: %.2f%%\n', svm_accuracy * 100);

%% ===================== Train KNN =====================
fprintf('Training KNN...\n');
knn_model = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5);
knn_pred = predict(knn_model, Xtest);
knn_accuracy = mean(knn_pred == Ytest);
fprintf('KNN Accuracy: %.2f%%\n', knn_accuracy * 100);

%% ===================== Train Random Forest =====================
fprintf('Training Random Forest...\n');
rf_model = TreeBagger(100, Xtrain, Ytrain, 'Method', 'classification');
rf_pred = str2double(predict(rf_model, Xtest));
rf_accuracy = mean(rf_pred == Ytest);
fprintf('Random Forest Accuracy: %.2f%%\n', rf_accuracy * 100);

%% ===================== Train Naive Bayes =====================
fprintf('Training Naive Bayes...\n');
nb_model = fitcnb(Xtrain, Ytrain);
nb_pred = predict(nb_model, Xtest);
nb_accuracy = mean(nb_pred == Ytest);
fprintf('Naive Bayes Accuracy: %.2f%%\n', nb_accuracy * 100);

%% ===================== Train Logistic Regression =====================
fprintf('Training Logistic Regression...\n');
lr_model = mnrfit(Xtrain, categorical(Ytrain));
lr_pred = double(mnrval(lr_model, Xtest) > 0.5);
lr_accuracy = mean(lr_pred == Ytest);
fprintf('Logistic Regression Accuracy: %.2f%%\n', lr_accuracy * 100);

%% ===================== Train 1D CNN =====================
fprintf('Training 1D CNN...\n');
Ytrain = Ytrain(:);
Ytest = Ytest(:);

numFeatures = size(Xtrain, 2);
numSamplesTrain = size(Xtrain, 1);
numSamplesTest = size(Xtest, 1);

Xtrain_cnn = reshape(Xtrain', [1, numFeatures, 1, numSamplesTrain]);
Xtest_cnn  = reshape(Xtest',  [1, numFeatures, 1, numSamplesTest]);

layers = [
    imageInputLayer([1 numFeatures 1], 'Normalization', 'none', 'Name', 'input')
    
    convolution2dLayer([1 3], 16, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    
    convolution2dLayer([1 3], 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    
    fullyConnectedLayer(1, 'Name', 'fc2')
    sigmoidLayer('Name', 'sigmoid')
    regressionLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ... % Changed to shuffle every epoch for better training
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the CNN model
cnn_model = trainNetwork(Xtrain_cnn, Ytrain, layers, options);

% Evaluate the CNN model on the test data
cnn_pred = predict(cnn_model, Xtest_cnn);
cnn_pred_label = double(cnn_pred > 0.5); 
cnn_accuracy = mean(cnn_pred_label == Ytest);
fprintf('1D CNN Accuracy: %.2f%%\n', cnn_accuracy * 100);

%% ===================== Confusion Matrices =====================
% Confusion Matrix for SVM
figure;
confusionchart(Ytest, svm_pred);
title('SVM Confusion Matrix');

% Confusion Matrix for KNN
figure;
confusionchart(Ytest, knn_pred);
title('KNN Confusion Matrix');

% Confusion Matrix for Random Forest
figure;
confusionchart(Ytest, rf_pred);
title('Random Forest Confusion Matrix');

% Confusion Matrix for Naive Bayes
figure;
confusionchart(Ytest, nb_pred);
title('Naive Bayes Confusion Matrix');

% Confusion Matrix for Logistic Regression
figure;
confusionchart(Ytest, lr_pred);
title('Logistic Regression Confusion Matrix');

% Confusion Matrix for 1D CNN
figure;
confusionchart(Ytest, cnn_pred_label);
title('1D CNN Confusion Matrix');

fprintf('\n=== MODEL TRAINING COMPLETE ===\n');
