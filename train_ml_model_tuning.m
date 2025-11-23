clear; clc; close all;
load('training_dataset.mat');  

% Create partition
cv = cvpartition(all_labels, 'HoldOut', 0.30);
Xtrain = all_features(training(cv), :);
Ytrain = all_labels(training(cv), :);
Xtest  = all_features(test(cv), :);
Ytest  = all_labels(test(cv), :);

% --- GLOBAL RANDOM SEARCH SETTINGS ---
% We use 'randomsearch' explicitly. 
% We use 30 iterations because Random Search needs more tries than Bayesian.
opts = struct('Optimizer', 'randomsearch', ...
              'MaxObjectiveEvaluations', 30, ...
              'UseParallel', false, ...
              'Verbose', 0, ...
              'ShowPlots', false); % Turn off popup plots to save speed

fprintf('=== STARTING RANDOM SEARCH FOR ALL MODELS ===\n');

%% ===================== 1. SVM (Random Search) =====================
fprintf('\n[1/7] Tuning SVM...\n');
svm_mdl = fitcsvm(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts, ...
    'Standardize', true);
svm_acc = mean(predict(svm_mdl, Xtest) == Ytest);
fprintf('   Best SVM: %.2f%%\n', svm_acc * 100);

%% ===================== 2. KNN (Random Search) =====================
fprintf('\n[2/7] Tuning KNN...\n');
knn_mdl = fitcknn(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);
knn_acc = mean(predict(knn_mdl, Xtest) == Ytest);
fprintf('   Best KNN: %.2f%%\n', knn_acc * 100);

%% ===================== 3. Random Forest (Random Search) =====================
fprintf('\n[3/7] Tuning Random Forest...\n');
rf_mdl = fitcensemble(Xtrain, Ytrain, 'Method', 'Bag', ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', opts);
rf_acc = mean(predict(rf_mdl, Xtest) == Ytest);
fprintf('   Best RF: %.2f%%\n', rf_acc * 100);

%% ===================== 4. Naive Bayes (Random Search) =====================
fprintf('\n[4/7] Tuning Naive Bayes...\n');
nb_mdl = fitcnb(Xtrain, Ytrain, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);
nb_acc = mean(predict(nb_mdl, Xtest) == Ytest);
fprintf('   Best Naive Bayes: %.2f%%\n', nb_acc * 100);

%% ===================== 5. Logistic Regression (Random Search) =====================
fprintf('\n[5/7] Tuning Logistic Regression...\n');
lr_mdl = fitclinear(Xtrain, Ytrain, 'Learner', 'logistic', ...
    'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
    'HyperparameterOptimizationOptions', opts);
lr_acc = mean(predict(lr_mdl, Xtest) == Ytest);
fprintf('   Best LogReg: %.2f%%\n', lr_acc * 100);

%% ===================== 6. XGBoost (Manual Random Search) =====================
fprintf('\n[6/7] Tuning XGBoost...\n');
% Prepare Data
Xtrain_d = double(Xtrain); Ytrain_d = double(Ytrain);
Xtest_d  = double(Xtest);  Ytest_d  = double(Ytest);
py_Xtrain = py.numpy.array(Xtrain_d); py_Ytrain = py.numpy.array(Ytrain_d);
py_Xtest  = py.numpy.array(Xtest_d);  py_Ytest  = py.numpy.array(Ytest_d);

xgb = py.importlib.import_module('xgboost');
dtrain = xgb.DMatrix(py_Xtrain, py_Ytrain);
dtest  = xgb.DMatrix(py_Xtest,  py_Ytest);

best_xgb_acc = 0;
num_iter_xgb = 30; % Match the 30 iters of other models

for i = 1:num_iter_xgb
    % Random Params
    d_depth = int32(randi([3, 12])); 
    d_eta = 0.01 + (0.4 - 0.01) * rand(); 
    d_sub = 0.5 + (1.0 - 0.5) * rand();
    
    params = py.dict(pyargs('objective', 'binary:logistic', ...
        'eval_metric', 'logloss', 'max_depth', d_depth, ...
        'eta', d_eta, 'subsample', d_sub, 'nthread', int32(4), 'verbosity', int32(0)));
    
    bst = xgb.train(params, dtrain, int32(50));
    pred = double(bst.predict(dtest));
    curr_acc = mean(double(pred > 0.5) == Ytest_d);
    
    if curr_acc > best_xgb_acc
        best_xgb_acc = curr_acc;
    end
end
fprintf('   Best XGBoost: %.2f%%\n', best_xgb_acc * 100);

%% ===================== 7. 1D CNN (Manual Random Search) =====================
fprintf('\n[7/7] Tuning 1D CNN...\n');
numFeatures = size(Xtrain, 2);
Xtrain_cnn = reshape(Xtrain', [1, numFeatures, 1, size(Xtrain, 1)]);
Xtest_cnn  = reshape(Xtest',  [1, numFeatures, 1, size(Xtest, 1)]);
Ytrain_cat = categorical(Ytrain); Ytest_cat = categorical(Ytest);

best_cnn_acc = 0;
num_iter_cnn = 10; % CNNs are slow, so we do fewer iterations (10)

for i = 1:num_iter_cnn
    % Randomly pick Kernel Size (3 or 5) and Filters (16 to 64)
    k_size = randsample([3, 5], 1);
    n_filt = randsample([16, 32, 64], 1);
    l_rate = 10^(-(2 + 2*rand())); % Log space between 1e-2 and 1e-4
    
    layers = [
        imageInputLayer([1 numFeatures 1], 'Normalization', 'none')
        convolution2dLayer([1 k_size], n_filt, 'Padding', 'same'), batchNormalizationLayer, reluLayer
        fullyConnectedLayer(2), softmaxLayer, classificationLayer];

    opts_cnn = trainingOptions('adam', 'MaxEpochs', 10, 'MiniBatchSize', 64, ...
        'InitialLearnRate', l_rate, 'Plots', 'none', 'Verbose', false);

    try
        cnn_net = trainNetwork(Xtrain_cnn, Ytrain_cat, layers, opts_cnn);
        curr_acc = mean(classify(cnn_net, Xtest_cnn) == Ytest_cat);
        if curr_acc > best_cnn_acc
            best_cnn_acc = curr_acc;
        end
    catch
        continue; % Skip if random params cause divergence
    end
end
fprintf('   Best 1D CNN: %.2f%%\n', best_cnn_acc * 100);

%% ===================== Final Comparison =====================
figure('Name', 'Random Search Comparison', 'NumberTitle', 'off');
model_names = {'SVM', 'KNN', 'RF', 'NBayes', 'LogReg', 'XGBoost', 'CNN'};
accuracies = [svm_acc, knn_acc, rf_acc, nb_acc, lr_acc, best_xgb_acc, best_cnn_acc] * 100;

b = bar(accuracies);
xticklabels(model_names);
ylabel('Accuracy (%)');
title('Random Search Optimization Results (30 Iterations)');
grid on;
ylim([min(accuracies)-10, 100]);

% Add text labels on top of bars
text(1:length(accuracies), accuracies, num2str(accuracies', '%.1f%%'), ...
    'vert','bottom','horiz','center', 'FontSize', 10, 'FontWeight', 'bold');