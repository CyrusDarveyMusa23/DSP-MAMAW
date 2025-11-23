clear; clc; close all;

feature_files = dir('features/*_features.mat');
all_features = [];
all_labels = [];

for i = 1:length(feature_files)
    file_path = fullfile(feature_files(i).folder, feature_files(i).name);
    fprintf('Processing file: %s\n', file_path);
    
    A = load(file_path);
    feature_data = A.feature_data;
    
    for s = 1:length(feature_data)
        feats = feature_data(s).features;
        labels = feature_data(s).labels;  
        
        all_features = [all_features; feats];
        all_labels = [all_labels; labels];
    end
end

fprintf('Dataset size: %d windows x %d features\n', size(all_features, 1), size(all_features, 2));

save('training_dataset.mat', 'all_features', 'all_labels');
