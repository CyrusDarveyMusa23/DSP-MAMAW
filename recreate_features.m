%% recreate_features.m - Complete feature recreation
function recreate_features()
% Recreate features from relabeled data

clear; clc; close all;

fprintf('=== RECREATING FEATURES FROM RELABELED DATA ===\n');

% Get all relabeled files
relabeled_files = dir('relabeled_data/*_relabeled.mat');

if isempty(relabeled_files)
    error('No relabeled files found! Run the labeling script first.');
end

fprintf('Found %d relabeled files:\n', length(relabeled_files));
for i = 1:length(relabeled_files)
    fprintf('  %d. %s\n', i, relabeled_files(i).name);
end

% Create features folder if it doesn't exist
if ~exist('features_recreated', 'dir')
    mkdir('features_recreated');
end

%% Process each relabeled file
for file_idx = 1:length(relabeled_files)
    file_path = fullfile(relabeled_files(file_idx).folder, relabeled_files(file_idx).name);
    fprintf('\nProcessing: %s\n', relabeled_files(file_idx).name);
    
    % Load relabeled data
    A = load(file_path);
    labeled_data = A.labeled_data;
    
    feature_data = struct();
    
    total_windows = 0;
    clean_windows = 0;
    artifact_windows = 0;
    
    for s = 1:length(labeled_data)
        fprintf('  Subject %d: ', s);
        
        ppg_windows = labeled_data(s).ppg_windows;
        acc_windows = labeled_data(s).acc_windows;
        labels = labeled_data(s).labels;
        fs = 64;  % PPG sampling rate
        
        numWins = size(ppg_windows, 1);
        feat_matrix = zeros(numWins, 16);
        
        % Extract features for each window
        for w = 1:numWins
            ppg_win = ppg_windows(w, :);
            acc_win = acc_windows(w, :);
            
            feat_matrix(w, :) = extract_ppg_features(ppg_win, acc_win, fs);
        end
        
        % Store feature data
        feature_data(s).features = feat_matrix;
        feature_data(s).labels = labels;
        
        total_windows = total_windows + numWins;
        clean_windows = clean_windows + sum(labels == 0);
        artifact_windows = artifact_windows + sum(labels == 1);
        
        fprintf('%d windows (%d clean, %d artifact)\n', ...
            numWins, sum(labels==0), sum(labels==1));
    end
    
    % Save features
    [~, name, ~] = fileparts(relabeled_files(file_idx).name);
    % Remove '_relabeled' from the name
    name = strrep(name, '_relabeled', '');
    output_file = fullfile('features_recreated', sprintf('%s_features.mat', name));
    
    save(output_file, 'feature_data');
    fprintf('Saved: %s\n', output_file);
end

fprintf('\n=== FEATURE RECREATION SUMMARY ===\n');
fprintf('Total windows processed: %d\n', total_windows);
fprintf('Clean windows: %d (%.1f%%)\n', clean_windows, clean_windows/total_windows*100);
fprintf('Artifact windows: %d (%.1f%%)\n', artifact_windows, artifact_windows/total_windows*100);
fprintf('\n=== FEATURE RECREATION COMPLETE ===\n');

end