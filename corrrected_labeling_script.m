%% CORRECTED labeling script - save as relabel_data.m
function relabel_data()
% Relabel your data with proper artifact detection

clear; clc; close all;

% Get all windowed files
windowed_files = dir('windowed_data/*_windowed.mat');

fprintf('Relabeling %d windowed files...\n', length(windowed_files));

for file_idx = 1:length(windowed_files)
    file_path = fullfile(windowed_files(file_idx).folder, windowed_files(file_idx).name);
    fprintf('\nProcessing: %s\n', windowed_files(file_idx).name);
    
    % Load windowed data
    A = load(file_path);
    windowed_data = A.windowed_data;
    
    labeled_data = struct();
    
    for s = 1:length(windowed_data)
        fprintf('  Subject %d: ', s);
        
        ppg_windows = windowed_data(s).ppg_windows;
        acc_windows = windowed_data(s).acc_windows;
        
        num_windows = size(ppg_windows, 1);
        labels = zeros(num_windows, 1); % Start with all clean
        
        % Label based on motion artifacts
        for w = 1:num_windows
            ppg_win = ppg_windows(w, :);
            acc_win = acc_windows(w, :);
            
            % Calculate artifact indicators
            acc_std = std(acc_win);
            ppg_std = std(ppg_win);
            correlation = corr(ppg_win(:), acc_win(:));
            
            % Simple rule-based labeling (you can adjust these thresholds)
            if acc_std > 100 || ppg_std > 150 || abs(correlation) > 0.6
                labels(w) = 1; % Artifact
            else
                labels(w) = 0; % Clean
            end
        end
        
        labeled_data(s).ppg_windows = ppg_windows;
        labeled_data(s).acc_windows = acc_windows;
        labeled_data(s).labels = labels;
        labeled_data(s).win_times = windowed_data(s).win_times;
        
        fprintf('%d windows (%d clean, %d artifact)\n', ...
            num_windows, sum(labels==0), sum(labels==1));
    end
    
    % Save relabeled data
    [~, name, ~] = fileparts(windowed_files(file_idx).name);
    output_file = fullfile('relabeled_data', sprintf('%s_relabeled.mat', name));
    
    if ~exist('relabeled_data', 'dir')
        mkdir('relabeled_data');
    end
    
    save(output_file, 'labeled_data');
    fprintf('Saved: %s\n', output_file);
end

fprintf('\n=== RELABELING COMPLETE ===\n');

end