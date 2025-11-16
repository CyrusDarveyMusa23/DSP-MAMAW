%% fix_data_leakage.m
clear; clc;

% Load your current dataset
load('training_dataset_new.mat');

% Remove motion-based features that are causing data leakage
% Feature 7: Mean ACC, Feature 8: Std ACC, Feature 9: Max ACC
% Feature 10: PPG-ACC Correlation, Feature 11: SQI
features_to_remove = [7, 8, 9, 10, 11];
clean_features = all_features;
clean_features(:, features_to_remove) = [];

fprintf('=== FIXING DATA LEAKAGE ===\n');
fprintf('Original features: %d\n', size(all_features, 2));
fprintf('Clean features: %d\n', size(clean_features, 2));
fprintf('Removed motion-based features that were cheating\n');

% Save the clean dataset
save('training_dataset_clean.mat', 'clean_features', 'all_labels');

% Check the improvement
fprintf('New max correlation with labels: %.3f\n', max(abs(corr(clean_features, all_labels))));
fprintf('Saved: training_dataset_clean.mat\n');
fprintf('\nNow run your training with the CLEAN dataset!\n');