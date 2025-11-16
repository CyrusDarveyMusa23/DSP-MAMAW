%% DEBUG: Check labeling process
clear; clc; close all;

% Check your labeled files
labeled_files = dir('labeled_data/*_labeled.mat');

fprintf('=== CHECKING LABELED FILES ===\n');

for i = 1:length(labeled_files)
    file_path = fullfile(labeled_files(i).folder, labeled_files(i).name);
    fprintf('\nFile: %s\n', labeled_files(i).name);
    
    A = load(file_path);
    labeled_data = A.labeled_data;
    
    total_windows = 0;
    clean_windows = 0;
    artifact_windows = 0;
    
    for s = 1:length(labeled_data)
        if isfield(labeled_data(s), 'labels')
            labels = labeled_data(s).labels;
            total_windows = total_windows + length(labels);
            clean_windows = clean_windows + sum(labels == 0);
            artifact_windows = artifact_windows + sum(labels == 1);
        end
    end
    
    fprintf('  Total windows: %d\n', total_windows);
    fprintf('  Clean windows: %d (%.1f%%)\n', clean_windows, clean_windows/total_windows*100);
    fprintf('  Artifact windows: %d (%.1f%%)\n', artifact_windows, artifact_windows/total_windows*100);
end

% Check one file in detail
if ~isempty(labeled_files)
    file_path = fullfile(labeled_files(1).folder, labeled_files(1).name);
    A = load(file_path);
    labeled_data = A.labeled_data;
    
    fprintf('\n=== DETAILED CHECK OF FIRST SUBJECT ===\n');
    fprintf('Number of subjects: %d\n', length(labeled_data));
    
    if length(labeled_data) >= 1
        subj_labels = labeled_data(1).labels;
        fprintf('Labels for subject 1:\n');
        fprintf('  Total: %d\n', length(subj_labels));
        fprintf('  Unique values: %s\n', mat2str(unique(subj_labels)));
        fprintf('  All zeros: %d\n', all(subj_labels == 0));
        fprintf('  All ones: %d\n', all(subj_labels == 1));
    end
end