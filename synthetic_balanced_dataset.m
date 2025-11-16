%% QUICK TEST: Create balanced dataset for testing
clear; clc; close all;

load('training_dataset.mat');

fprintf('Original dataset: %d samples, %.1f%% artifacts\n', ...
    size(all_features, 1), mean(all_labels)*100);

% Since all labels are 1, let's create some clean samples
% We'll randomly select half the samples and label them as clean
rng(42);
n_samples = size(all_features, 1);
new_labels = ones(n_samples, 1); % Start with all artifacts

% Randomly select 50% to be clean
clean_indices = randperm(n_samples, round(n_samples * 0.5));
new_labels(clean_indices) = 0;

fprintf('Synthetic dataset: %d samples, %.1f%% artifacts\n', ...
    n_samples, mean(new_labels)*100);

% Save synthetic dataset for testing
save('training_dataset_balanced.mat', 'all_features', 'new_labels');

% Now train with this balanced dataset
all_labels = new_labels;

% Continue with your training...
fprintf('\nNow run enhanced_ml_training with the balanced dataset\n');