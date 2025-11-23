clear; clc; close all;

% Load the 70-30 split dataset
load('training_dataset.mat');  % Contains: all_features, all_labels

% Create the 70-30 split using cvpartition
cv = cvpartition(all_labels, 'HoldOut', 0.30);
Xtrain = all_features(training(cv), :);
Ytrain = all_labels(training(cv), :);
Xtest  = all_features(test(cv), :);
Ytest  = all_labels(test(cv), :);

numFeatures = size(Xtrain, 2);

%% ===================== 1. SVM =====================
fprintf('\n[1/6] Evaluating SVM...\n');
svm_model = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'Standardize', true);
svm_pred = predict(svm_model, Xtest);
[~, svm_scores] = predict(svm_model, Xtest); 

svm_acc = mean(svm_pred == Ytest);
svm_cm = confusionmat(Ytest, svm_pred);
[svm_prec, svm_rec, svm_f1] = calculate_metrics(svm_cm);
fprintf('   SVM Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    svm_acc*100, svm_prec*100, svm_rec*100, svm_f1);

plot_roc_curve(Ytest, svm_scores(:,2), 'SVM');

%% ===================== 2. KNN =====================
fprintf('\n[2/6] Evaluating KNN...\n');
knn_model = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 7, 'Distance', 'cityblock', 'Standardize', true);
knn_pred = predict(knn_model, Xtest);
[~, knn_scores] = predict(knn_model, Xtest);

knn_acc = mean(knn_pred == Ytest);
knn_cm = confusionmat(Ytest, knn_pred);
[knn_prec, knn_rec, knn_f1] = calculate_metrics(knn_cm);
fprintf('   KNN Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    knn_acc*100, knn_prec*100, knn_rec*100, knn_f1);

plot_roc_curve(Ytest, knn_scores(:,2), 'KNN');

%% ===================== 3. Random Forest =====================
fprintf('\n[3/6] Evaluating Random Forest...\n');
rf_model = TreeBagger(100, Xtrain, Ytrain, 'Method', 'classification');
rf_pred_cell = predict(rf_model, Xtest);
rf_pred = str2double(rf_pred_cell);
[~, rf_scores] = predict(rf_model, Xtest);

rf_acc = mean(rf_pred == Ytest);
rf_cm = confusionmat(Ytest, rf_pred);
[rf_prec, rf_rec, rf_f1] = calculate_metrics(rf_cm);
fprintf('   RF Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    rf_acc*100, rf_prec*100, rf_rec*100, rf_f1);

plot_roc_curve(Ytest, rf_scores(:,2), 'Random Forest');

%% ===================== 4. Naive Bayes =====================
fprintf('\n[4/6] Evaluating Naive Bayes...\n');
nb_model = fitcnb(Xtrain, Ytrain, 'Distribution', 'Kernel', 'Width', 0.0010002);
nb_pred = predict(nb_model, Xtest);
[~, nb_scores] = predict(nb_model, Xtest);

nb_acc = mean(nb_pred == Ytest);
nb_cm = confusionmat(Ytest, nb_pred);
[nb_prec, nb_rec, nb_f1] = calculate_metrics(nb_cm);
fprintf('   NB Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    nb_acc*100, nb_prec*100, nb_rec*100, nb_f1);

plot_roc_curve(Ytest, nb_scores(:,2), 'Naive Bayes');

%% ===================== 5. Logistic Regression =====================
fprintf('\n[5/6] Evaluating Logistic Regression...\n');
lr_beta = mnrfit(Xtrain, categorical(Ytrain));
lr_probs = mnrval(lr_beta, Xtest); 
lr_scores = lr_probs(:,2);
lr_pred = double(lr_scores > 0.5);

lr_acc = mean(lr_pred == Ytest);
lr_cm = confusionmat(Ytest, lr_pred);
[lr_prec, lr_rec, lr_f1] = calculate_metrics(lr_cm);
fprintf('   LogReg Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    lr_acc*100, lr_prec*100, lr_rec*100, lr_f1);

plot_roc_curve(Ytest, lr_scores, 'Logistic Regression');

%% ===================== 6. 1D CNN =====================
fprintf('\n[6/6] Evaluating 1D CNN...\n');
Xtrain_cnn = reshape(Xtrain', [1, numFeatures, 1, size(Xtrain, 1)]);
Xtest_cnn  = reshape(Xtest',  [1, numFeatures, 1, size(Xtest, 1)]);
Ytrain_cat = categorical(Ytrain); 
Ytest_cat  = categorical(Ytest);

layers = [
    imageInputLayer([1 numFeatures 1], 'Normalization', 'none')
    
    convolution2dLayer([1 3], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([1 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', 'MaxEpochs', 15, 'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, 'Plots', 'none', 'Verbose', false);

cnn_net = trainNetwork(Xtrain_cnn, Ytrain_cat, layers, options);
[cnn_pred_cat, cnn_scores] = classify(cnn_net, Xtest_cnn);
cnn_pred = double(cnn_pred_cat) - 1; 

cnn_acc = mean(cnn_pred == Ytest);
cnn_cm = confusionmat(Ytest, cnn_pred);
[cnn_prec, cnn_rec, cnn_f1] = calculate_metrics(cnn_cm);
fprintf('   CNN Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.4f\n', ...
    cnn_acc*100, cnn_prec*100, cnn_rec*100, cnn_f1);

plot_roc_curve(Ytest, cnn_scores(:,2), '1D CNN');

%% ===================== Comparison Chart =====================
figure('Name', 'Model Comparison', 'NumberTitle', 'off');
model_names = {'SVM', 'KNN', 'RF', 'NB', 'LogReg', 'CNN'};
accuracies = [svm_acc, knn_acc, rf_acc, nb_acc, lr_acc, cnn_acc] * 100;

b = bar(accuracies);
xticklabels(model_names);
ylabel('Accuracy (%)');
title('Model Accuracy Comparison');
grid on;
ylim([min(accuracies)-10, 100]);
text(1:length(accuracies), accuracies, num2str(accuracies', '%.1f%%'), ...
    'vert','bottom','horiz','center', 'FontSize', 10, 'FontWeight', 'bold');

fprintf('\n=== MODEL EVALUATION COMPLETE ===\n');

%% ===================== HELPER FUNCTIONS =====================

function [precision, recall, f1_score] = calculate_metrics(conf_matrix)
    if size(conf_matrix,1) < 2
        precision = 0; recall = 0; f1_score = 0;
        return;
    end

    TP = conf_matrix(2, 2);  
    FP = conf_matrix(1, 2);  
    FN = conf_matrix(2, 1);  
    TN = conf_matrix(1, 1);  
    
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    if isnan(precision), precision = 0; end
    if isnan(recall), recall = 0; end
    if isnan(f1_score), f1_score = 0; end
end

function plot_roc_curve(true_labels, predicted_scores, modelName)
    [X, Y, T, AUC] = perfcurve(true_labels, predicted_scores, 1);
    figure;
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title([modelName ' ROC Curve (AUC = ' num2str(AUC, '%.3f') ')']);
    grid on;
end

%% ===================== Export the Best Model =====================
best_model = rf_model;  
output_file = 'best_model_random_forest.mat';
save(output_file, 'best_model');
fprintf('Best model saved to: %s\n', output_file);
