%% enhanced_ml_training_complete.m - WITH ALL 7 MODELS & DATA LEAKAGE FIX
function enhanced_ml_training_complete()
% Enhanced ML Training with ALL 7 classifiers

clear; clc; close all;

%% Load the prepared dataset - UPDATED TO USE CLEAN DATA
if exist('training_dataset_clean.mat', 'file')
    load('training_dataset_clean.mat');
    fprintf('Using CLEAN dataset (motion features removed)\n');
    all_features = clean_features; % Use the cleaned features
elseif exist('training_dataset_new.mat', 'file')
    load('training_dataset_new.mat');
    fprintf('Using NEW balanced dataset\n');
elseif exist('training_dataset.mat', 'file')
    load('training_dataset.mat');
    fprintf('Using original dataset\n');
else
    error('No training dataset found! Run prepare_training_dataset first.');
end

fprintf('Dataset loaded: %d samples, %d features\n', size(all_features, 1), size(all_features, 2));

%% Check dataset quality
fprintf('\n=== DATASET QUALITY CHECK ===\n');
fprintf('Total samples: %d\n', size(all_features, 1));
fprintf('Class distribution: %.1f%% clean, %.1f%% artifact\n', ...
    mean(all_labels==0)*100, mean(all_labels==1)*100);

% Check for data leakage in features
max_correlation = max(abs(corr(all_features, all_labels)));
fprintf('Max feature-label correlation: %.3f\n', max_correlation);
if max_correlation > 0.8
    fprintf('⚠️  Warning: High correlation detected - may indicate data leakage\n');
end

if all(all_labels == 0) || all(all_labels == 1)
    fprintf('🚨 CRITICAL: Dataset contains only ONE class!\n');
    return;
end

%% Use BALANCED split
fprintf('\n=== CREATING BALANCED TRAIN/TEST SPLIT ===\n');
[Xtrain, Ytrain, Xtest, Ytest] = balanced_split(all_features, all_labels, 0.30);

fprintf('Training set: %d samples (%.1f%% clean, %.1f%% artifact)\n', ...
    size(Xtrain, 1), mean(Ytrain==0)*100, mean(Ytrain==1)*100);
fprintf('Test set:     %d samples (%.1f%% clean, %.1f%% artifact)\n', ...
    size(Xtest, 1), mean(Ytest==0)*100, mean(Ytest==1)*100);

%% ============= TRAIN ALL 7 MODELS =================
fprintf('\n=== TRAINING ALL 7 MODELS ===\n');

models = struct();

% 1. SVM
fprintf('1. Training SVM...\n');
try
    models.svm = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', ...
        'Standardize', true, 'BoxConstraint', 1);
    fprintf('   ✓ SVM training completed\n');
catch ME
    fprintf('   ✗ SVM training failed: %s\n', ME.message);
end

% 2. Random Forest
fprintf('2. Training Random Forest...\n');
try
    models.rf = TreeBagger(100, Xtrain, Ytrain, 'Method', 'classification', ...
        'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', 'MinLeafSize', 5);
    fprintf('   ✓ Random Forest training completed\n');
catch ME
    fprintf('   ✗ Random Forest training failed: %s\n', ME.message);
end

% 3. KNN
fprintf('3. Training KNN...\n');
try
    models.knn = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5, 'Standardize', true);
    fprintf('   ✓ KNN training completed\n');
catch ME
    fprintf('   ✗ KNN training failed: %s\n', ME.message);
end

% 4. Naive Bayes
fprintf('4. Training Naive Bayes...\n');
try
    models.nb = fitcnb(Xtrain, Ytrain, 'DistributionNames', 'normal');
    fprintf('   ✓ Naive Bayes training completed\n');
catch ME
    fprintf('   ✗ Naive Bayes training failed: %s\n', ME.message);
end

% 5. Logistic Regression
fprintf('5. Training Logistic Regression...\n');
try
    models.lr = fitclinear(Xtrain, Ytrain, 'Learner', 'logistic', ...
        'Lambda', 0.01, 'Regularization', 'ridge');
    fprintf('   ✓ Logistic Regression training completed\n');
catch ME
    fprintf('   ✗ Logistic Regression training failed: %s\n', ME.message);
end

% 6. XGBoost (MATLAB equivalent) - UPDATED WITH REGULARIZATION
fprintf('6. Training XGBoost (Gradient Boosting)...\n');
try
    models.xgboost = fitcensemble(Xtrain, Ytrain, 'Method', 'GentleBoost', ...
        'NumLearningCycles', 100, 'LearnRate', 0.05, 'Learners', 'tree'); % Reduced complexity
    fprintf('   ✓ XGBoost training completed\n');
catch ME
    fprintf('   ✗ XGBoost training failed: %s\n', ME.message);
end

% 7. 1D-CNN
fprintf('7. Training 1D-CNN...\n');
try
    [models.cnn, cnn_success] = train_1d_cnn_simple(Xtrain, Ytrain, Xtest, Ytest);
    if cnn_success
        fprintf('   ✓ 1D-CNN training completed\n');
    else
        fprintf('   ✗ 1D-CNN training failed\n');
    end
catch ME
    fprintf('   ✗ 1D-CNN training failed: %s\n', ME.message);
end

%% ============= MAKE PREDICTIONS =================
fprintf('\n=== MAKING PREDICTIONS ===\n');

predictions = struct();
model_names = fieldnames(models);

% FIX: Remove empty CNN model if it exists
if isfield(models, 'cnn') && isempty(models.cnn)
    models = rmfield(models, 'cnn');
    model_names = fieldnames(models); % Update model names list
    fprintf('Removed empty CNN model\n');
end

for i = 1:length(model_names)
    model_name = model_names{i};
    fprintf('Predicting with %s... ', model_name);
    
    try
        if strcmp(model_name, 'rf')
            % Random Forest
            pred = str2double(predict(models.(model_name), Xtest));
        elseif strcmp(model_name, 'cnn')
            % 1D-CNN - FIXED: Check if model is valid
            if ~isempty(models.cnn)
                Xtest_cnn = reshape(Xtest', [1, 1, size(Xtest, 2), size(Xtest, 1)]);
                cnn_pred = classify(models.cnn, Xtest_cnn);
                pred = double(cnn_pred) - 1; % Convert to 0/1
            else
                pred = [];
                fprintf('SKIP (empty model) ');
            end
        else
            % Other models
            pred = predict(models.(model_name), Xtest);
        end
        
        if ~isempty(pred)
            predictions.(model_name) = pred;
            fprintf('✓\n');
        else
            predictions.(model_name) = [];
            fprintf('✗ (no predictions)\n');
        end
        
    catch ME
        fprintf('✗ Error: %s\n', ME.message);
        predictions.(model_name) = [];
    end
end

%% ============= CALCULATE PERFORMANCE METRICS =================
fprintf('\n=== MODEL PERFORMANCE ===\n');

results = struct();

for i = 1:length(model_names)
    model_name = model_names{i};
    
    if isempty(predictions.(model_name))
        continue;
    end
    
    y_pred = predictions.(model_name);
    
    % Basic metrics
    accuracy = mean(y_pred == Ytest);
    
    % Confusion matrix
    cm = confusionmat(Ytest, y_pred);
    if size(cm, 1) == 2
        TP = cm(2,2);
        TN = cm(1,1);
        FP = cm(1,2);
        FN = cm(2,1);
        
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1 = 2 * (precision * recall) / (precision + recall + eps);
        specificity = TN / (TN + FP + eps);
    else
        precision = NaN;
        recall = NaN;
        f1 = NaN;
        specificity = NaN;
    end
    
    % Store results
    results.(model_name).accuracy = accuracy;
    results.(model_name).precision = precision;
    results.(model_name).recall = recall;
    results.(model_name).f1 = f1;
    results.(model_name).specificity = specificity;
    
    % Display
    fprintf('\n%s:\n', upper(model_name));
    fprintf('  Accuracy:    %.2f%%\n', accuracy*100);
    fprintf('  Precision:   %.3f\n', precision);
    fprintf('  Recall:      %.3f\n', recall);
    fprintf('  F1-Score:    %.3f\n', f1);
    fprintf('  Specificity: %.3f\n', specificity);
end

%% ============= VISUALIZE RESULTS =================
fprintf('\n=== CREATING VISUALIZATIONS ===\n');
plot_comprehensive_results(results, Ytest, predictions, model_names);

%% ============= SAVE RESULTS =================
fprintf('\n=== SAVING RESULTS ===\n');

% Create results folder
if ~exist('results', 'dir')
    mkdir('results');
end

% Save models and results
save('results/trained_models_complete.mat', 'models', 'results', 'predictions');
fprintf('Saved: results/trained_models_complete.mat\n');

% Save performance summary
save_performance_summary(results, model_names);

% Find and save best model
find_and_save_best_model(models, results, model_names);

fprintf('\n=== TRAINING COMPLETE ===\n');
fprintf('All 7 models trained and saved!\n');

end

%% ============= SUPPORT FUNCTIONS =================

function [Xtrain, Ytrain, Xtest, Ytest] = balanced_split(features, labels, test_ratio)
    % Get indices of each class
    clean_idx = find(labels == 0);
    artifact_idx = find(labels == 1);
    
    % Calculate how many from each class for test set
    n_test_clean = round(length(clean_idx) * test_ratio);
    n_test_artifact = round(length(artifact_idx) * test_ratio);
    
    % Randomly select test samples from each class
    rng(42);
    test_clean = randsample(clean_idx, n_test_clean);
    test_artifact = randsample(artifact_idx, n_test_artifact);
    
    % Combine test indices
    test_idx = [test_clean; test_artifact];
    train_idx = setdiff(1:length(labels), test_idx);
    
    % Create splits
    Xtrain = features(train_idx, :);
    Ytrain = labels(train_idx);
    Xtest = features(test_idx, :);
    Ytest = labels(test_idx);
end

function [cnn_net, success] = train_1d_cnn_simple(Xtrain, Ytrain, ~, ~)
    success = false;
    cnn_net = [];
    
    try
        % Check if Deep Learning Toolbox is available
        if ~license('test', 'Neural_Network_Toolbox')
            fprintf('   Deep Learning Toolbox not available - skipping CNN\n');
            return;
        end
        
        % Additional check: Verify specific functions exist
        if ~exist('imageInputLayer', 'file')
            fprintf('   Deep Learning Toolbox functions not found - skipping CNN\n');
            return;
        end
        
        % Reshape data for CNN
        Xtrain_cnn = reshape(Xtrain', [1, 1, size(Xtrain, 2), size(Xtrain, 1)]);
        Ytrain_cat = categorical(Ytrain);
        
        % Simple 1D-CNN architecture
        layers = [
            imageInputLayer([1 1 size(Xtrain, 2)])
            convolution2dLayer([1 3], 32, 'Padding', 'same')
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer
        ];
        
        % Training options
        options = trainingOptions('adam', ...
            'MaxEpochs', 20, ...
            'InitialLearnRate', 0.001, ...
            'Verbose', false, ...
            'Plots', 'none');
        
        % Train the network
        cnn_net = trainNetwork(Xtrain_cnn, Ytrain_cat, layers, options);
        success = true;
        fprintf('   ✓ 1D-CNN training completed\n');
        
    catch ME
        fprintf('   ✗ CNN training failed: %s\n', ME.message);
        success = false;
        cnn_net = [];
    end
end

function plot_comprehensive_results(results, Ytest, predictions, model_names)
    figure('Position', [100, 100, 1400, 800]);
    
    % Filter valid models
    valid_models = {};
    accuracies = [];
    f1_scores = [];
    
    for i = 1:length(model_names)
        if ~isempty(predictions.(model_names{i}))
            valid_models{end+1} = model_names{i};
            accuracies(end+1) = results.(model_names{i}).accuracy;
            f1_scores(end+1) = results.(model_names{i}).f1;
        end
    end
    
    % Accuracy comparison
    subplot(2,3,1);
    bar(accuracies * 100);
    set(gca, 'XTickLabel', valid_models, 'XTickLabelRotation', 45);
    ylabel('Accuracy (%)');
    title('Model Accuracy Comparison');
    grid on;
    
    % F1-Score comparison
    subplot(2,3,2);
    bar(f1_scores);
    set(gca, 'XTickLabel', valid_models, 'XTickLabelRotation', 45);
    ylabel('F1-Score');
    title('F1-Score Comparison');
    grid on;
    
    % Confusion matrices for top 2 models
    if length(valid_models) >= 2
        [~, idx] = sort(f1_scores, 'descend');
        
        subplot(2,3,3);
        confusionchart(Ytest, predictions.(valid_models{idx(1)}));
        title(sprintf('%s (F1=%.3f)', valid_models{idx(1)}, f1_scores(idx(1))));
        
        subplot(2,3,4);
        confusionchart(Ytest, predictions.(valid_models{idx(2)}));
        title(sprintf('%s (F1=%.3f)', valid_models{idx(2)}, f1_scores(idx(2))));
    end
    
    % Model ranking
    subplot(2,3,5);
    [sorted_f1, sort_idx] = sort(f1_scores, 'descend');
    barh(sorted_f1);
    set(gca, 'YTickLabel', valid_models(sort_idx));
    xlabel('F1-Score');
    title('Models Ranked by F1-Score');
    grid on;
    
    % Class distribution
    subplot(2,3,6);
    pie([sum(Ytest==0), sum(Ytest==1)], {'Clean', 'Artifact'});
    title('Test Set Distribution');
    
    sgtitle('7-Model Performance Comparison (Data Leakage Fixed)');
end

function save_performance_summary(results, model_names)
    performance_table = table();
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if ~isfield(results, model_name) || isempty(results.(model_name).accuracy)
            continue;
        end
        
        % Create a new row
        new_row = table();
        new_row.Model = {model_name};
        new_row.Accuracy = results.(model_name).accuracy * 100;
        new_row.Precision = results.(model_name).precision;
        new_row.Recall = results.(model_name).recall;
        new_row.F1_Score = results.(model_name).f1;
        new_row.Specificity = results.(model_name).specificity;
        
        % Append the new row
        if height(performance_table) == 0
            performance_table = new_row;
        else
            performance_table = [performance_table; new_row];
        end
    end
    
    writetable(performance_table, 'results/performance_summary_complete.csv');
    fprintf('Saved: results/performance_summary_complete.csv\n');
end

function find_and_save_best_model(models, results, model_names)
    % Find best model by F1-score
    best_f1 = -1;
    best_model_name = '';
    best_model = [];
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if isfield(results, model_name) && ~isempty(results.(model_name).f1)
            if results.(model_name).f1 > best_f1
                best_f1 = results.(model_name).f1;
                best_model_name = model_name;
                best_model = models.(model_name);
            end
        end
    end
    
    if ~isempty(best_model)
        best_model_data = struct();
        best_model_data.model = best_model;
        best_model_data.model_name = best_model_name;
        best_model_data.performance = results.(best_model_name);
        % UPDATED: Feature names without motion features
        best_model_data.feature_names = {'Mean PPG', 'Std PPG', 'Skewness', 'Kurtosis', ...
            'Dom Freq', 'Spectral Entropy', 'RMS PPG', 'Median PPG', 'IQR PPG', 'ZC Rate', 'Peak-to-Peak'};
        
        save('results/best_model_complete.mat', 'best_model_data');
        fprintf('Best model: %s (F1: %.3f)\n', best_model_name, best_f1);
        fprintf('Saved: results/best_model_complete.mat\n');
    end
end