%% create_new_training_dataset.m
function create_new_training_dataset()
% Create new training dataset from recreated features

clear; clc; close all;

fprintf('=== CREATING NEW TRAINING DATASET ===\n');

% Get all recreated feature files
feature_files = dir('features_recreated/*_features.mat');

if isempty(feature_files)
    error('No recreated feature files found! Run recreate_features first.');
end

fprintf('Found %d feature files:\n', length(feature_files));

all_features = [];
all_labels   = [];

% Combine all data
for i = 1:length(feature_files)
    file_path = fullfile(feature_files(i).folder, feature_files(i).name);
    fprintf('Loading: %s\n', feature_files(i).name);
    
    A = load(file_path);
    feature_data = A.feature_data;
    
    for s = 1:length(feature_data)
        feats = feature_data(s).features;
        labels = feature_data(s).labels;
        
        all_features = [all_features; feats];
        all_labels   = [all_labels; labels];
    end
end

fprintf('\n=== NEW DATASET SUMMARY ===\n');
fprintf('Total windows: %d\n', size(all_features, 1));
fprintf('Features per window: %d\n', size(all_features, 2));
fprintf('Clean windows: %d (%.1f%%)\n', sum(all_labels==0), mean(all_labels==0)*100);
fprintf('Artifact windows: %d (%.1f%%)\n', sum(all_labels==1), mean(all_labels==1)*100);

% Check for any NaN or Inf values
fprintf('Any NaN in features: %d\n', sum(any(isnan(all_features), 2)));
fprintf('Any Inf in features: %d\n', sum(any(isinf(all_features), 2)));

% Remove any rows with NaN or Inf
valid_rows = ~any(isnan(all_features), 2) & ~any(isinf(all_features), 2);
all_features = all_features(valid_rows, :);
all_labels = all_labels(valid_rows);

fprintf('After cleaning: %d windows\n', size(all_features, 1));

% Save the new combined dataset
save('training_dataset_new.mat', 'all_features', 'all_labels');
fprintf('\nSaved: training_dataset_new.mat\n');

% Quick visualization
figure('Position', [100 100 1000 400]);

subplot(1,2,1);
pie([sum(all_labels==0), sum(all_labels==1)], {'Clean', 'Artifact'});
title('Class Distribution');

subplot(1,2,2);
feature_variance = var(all_features);
bar(feature_variance);
xlabel('Feature Index');
ylabel('Variance');
title('Feature Variance');
grid on;

sgtitle('New Dataset Overview');

fprintf('\n=== DATASET CREATION COMPLETE ===\n');

end