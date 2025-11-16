%% CRITICAL DIAGNOSTIC
clear; clc;
load('training_dataset_new.mat');

fprintf('=== CRITICAL DATA CHECK ===\n');

% 1. Check if any feature perfectly separates classes
fprintf('1. Checking for perfect predictors...\n');
for i = 1:size(all_features, 2)
    unique_vals = unique(all_features(:,i));
    if length(unique_vals) == 2 % Binary feature
        class_0_vals = unique(all_features(all_labels==0, i));
        class_1_vals = unique(all_features(all_labels==1, i));
        
        if isempty(intersect(class_0_vals, class_1_vals))
            fprintf('🚨 FEATURE %d PERFECTLY SEPARATES CLASSES!\n', i);
        end
    end
end

% 2. Check data splitting
fprintf('\n2. Checking train/test split...\n');
rng(42);
cv = cvpartition(all_labels, 'HoldOut', 0.3);
Xtrain = all_features(training(cv), :);
Xtest = all_features(test(cv), :);

if isequal(Xtrain, Xtest)
    fprintf('🚨 TRAIN AND TEST SETS ARE IDENTICAL!\n');
end

% 3. Check feature correlations with labels
fprintf('\n3. Feature-label correlations:\n');
correlations = zeros(1, size(all_features, 2));
for i = 1:size(all_features, 2)
    correlations(i) = corr(all_features(:,i), all_labels);
end
fprintf('Max correlation: %.3f\n', max(abs(correlations)));
fprintf('Features with |corr| > 0.9: %d\n', sum(abs(correlations) > 0.9));

% 4. Check if problem is trivial
fprintf('\n4. Baseline performance:\n');
majority_class_accuracy = max(mean(all_labels==0), mean(all_labels==1)) * 100;
fprintf('Accuracy if always predict majority class: %.1f%%\n', majority_class_accuracy);