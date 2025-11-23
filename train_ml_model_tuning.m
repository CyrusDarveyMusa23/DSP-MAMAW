clear; clc; close all;
load('training_dataset.mat');  

% Create partition
cv = cvpartition(all_labels, 'HoldOut', 0.30);
Xtrain = all_features(training(cv), :);
Ytrain = all_labels(training(cv), :);
Xtest  = all_features(test(cv), :);
Ytest  = all_labels(test(cv), :);

% --- GLOBAL SETTINGS ---
% Limit the optimizer to 15 iterations to save time. 
% Increase to 30 for better results if you can wait longer.
opts = struct('MaxObjectiveEvaluations', 15, 'UseParallel', true);

%% ===================== Bayesian Optimization: SVM =====================
fprintf('\n--- Tuning SVM (Bayesian Opt) ---\n');
% 'auto' tunes BoxConstraint, KernelScale, and KernelFunction
svm_mdl = fitcsvm(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts, ...
    'Standardize', true);

svm_pred = predict(svm_mdl, Xtest);
svm_acc = mean(svm_pred == Ytest);
fprintf('Best SVM Accuracy: %.2f%%\n', svm_acc * 100);

%% ===================== Bayesian Optimization: KNN =====================
fprintf('\n--- Tuning KNN (Bayesian Opt) ---\n');
% 'auto' tunes NumNeighbors and Distance
knn_mdl = fitcknn(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);

knn_pred = predict(knn_mdl, Xtest);
knn_acc = mean(knn_pred == Ytest);
fprintf('Best KNN Accuracy: %.2f%%\n', knn_acc * 100);

%% ===================== Bayesian Optimization: Random Forest =====================
fprintf('\n--- Tuning Random Forest (Bayesian Opt) ---\n');
rf_mdl = fitcensemble(Xtrain, Ytrain, ...
    'Method', 'Bag', ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
    'HyperparameterOptimizationOptions', opts);

rf_pred = predict(rf_mdl, Xtest);
rf_acc = mean(rf_pred == Ytest);
fprintf('Best Random Forest Accuracy: %.2f%%\n', rf_acc * 100);

%% ===================== Bayesian Optimization: Naive Bayes =====================
fprintf('\n--- Tuning Naive Bayes (Bayesian Opt) ---\n');
% 'auto' tunes Distribution and Width
nb_mdl = fitcnb(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);

nb_pred = predict(nb_mdl, Xtest);
nb_acc = mean(nb_pred == Ytest);
fprintf('Best Naive Bayes Accuracy: %.2f%%\n', nb_acc * 100);

%% ===================== Bayesian Optimization: Logistic Regression =====================
fprintf('\n--- Tuning Logistic Regression (Bayesian Opt) ---\n');
lr_mdl = fitclinear(Xtrain, Ytrain, ...
    'Learner', 'logistic', ...
    'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
    'HyperparameterOptimizationOptions', opts);

lr_pred = predict(lr_mdl, Xtest);
lr_acc = mean(lr_pred == Ytest);
fprintf('Best Logistic Regression Accuracy: %.2f%%\n', lr_acc * 100);

%% ===================== Manual: 1D CNN =====================
% CNNs in MATLAB don't support the simple 'OptimizeHyperparameters' flag.
% We stick to a simple manual setting here to save time, as Deep Learning
% bayesian opt requires a complex custom objective function file.
fprintf('\n--- Training 1D CNN (Fixed Params) ---\n');

numFeatures = size(Xtrain, 2);
% Reshape for CNN
Xtrain_cnn = reshape(Xtrain', [1, numFeatures, 1, size(Xtrain, 1)]);
Xtest_cnn  = reshape(Xtest',  [1, numFeatures, 1, size(Xtest, 1)]);
Ytrain_cat = categorical(Ytrain); 
Ytest_cat  = categorical(Ytest);

layers = [
    imageInputLayer([1 numFeatures 1], 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([1 3], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([1 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'Plots', 'none', ...
    'Verbose', false);

cnn_net = trainNetwork(Xtrain_cnn, Ytrain_cat, layers, options);
cnn_pred = classify(cnn_net, Xtest_cnn);
cnn_acc = mean(cnn_pred == Ytest_cat);
fprintf('1D CNN Accuracy: %.2f%%\n', cnn_acc * 100);

%% ===================== Comparison Chart =====================
figure;
model_names = {'SVM', 'KNN', 'RandForest', 'NaiveBayes', 'LogReg', 'CNN'};
accuracies = [svm_acc, knn_acc, rf_acc, nb_acc, lr_acc, cnn_acc] * 100;

bar(accuracies);
xticklabels(model_names);
ylabel('Accuracy (%)');
title('Bayesian Optimized Model Comparison');
grid on;
ylim([min(accuracies)-5, 100]);