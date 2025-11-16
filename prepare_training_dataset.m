%% prepare_training_dataset.m
clear; clc; close all;

feature_folder = 'features';

% Find all feature files
files = dir(fullfile(feature_folder, '*_features.mat'));

if isempty(files)
    error('No feature files found! Run feature extraction first.');
end

fprintf('Found %d feature files:\n', length(files));
for i = 1:length(files)
    fprintf('  %d. %s\n', i, files(i).name);
end

all_features = [];
all_labels   = [];

% Combine all data
for i = 1:length(files)
    file_path = fullfile(files(i).folder, files(i).name);
    fprintf('Loading: %s\n', files(i).name);
    
    A = load(file_path);
    feature_data = A.feature_data;
    
    for s = 1:length(feature_data)
        feats = feature_data(s).features;
        labels = feature_data(s).labels;
        
        all_features = [all_features; feats];
        all_labels   = [all_labels; labels];
    end
end

fprintf('\n=== DATASET SUMMARY ===\n');
fprintf('Total windows: %d\n', size(all_features, 1));
fprintf('Features per window: %d\n', size(all_features, 2));
fprintf('Clean windows: %d (%.1f%%)\n', sum(all_labels==0), mean(all_labels==0)*100);
fprintf('Artifact windows: %d (%.1f%%)\n', sum(all_labels==1), mean(all_labels==1)*100);

% Save the combined dataset
save('training_dataset.mat', 'all_features', 'all_labels');
fprintf('\nSaved: training_dataset.mat\n');