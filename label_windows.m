function relabel_data_improved()
    % Relabel your data with improved artifact detection
    clear; clc; close all;
    
    % === SETTINGS ===
    % Saturation: If signal is swinging wildly (adjust to your unit context)
    SATURATION_THRESH = 160; 
    % Flatline: If std is near zero, sensor is likely disconnected
    FLATLINE_THRESH = 1.0;   
    % Correlation: Strictness of artifact matching
    CORR_THRESH = 0.7;       
    
    % Get all windowed files
    windowed_files = dir('windowed_data/*_windowed.mat');
    fprintf('Relabeling %d windowed files...\n', length(windowed_files));
    
    if isempty(windowed_files)
        error('No files found in windowed_data folder.');
    end

    for file_idx = 1:length(windowed_files)
        file_path = fullfile(windowed_files(file_idx).folder, windowed_files(file_idx).name);
        fprintf('\n------------------------------------------------\n');
        fprintf('Processing File %d/%d: %s\n', file_idx, length(windowed_files), windowed_files(file_idx).name);
        
        % Load windowed data
        A = load(file_path);
        windowed_data = A.windowed_data;
        
        labeled_data = struct();
        
        % 1. Global Histogram Analysis for this File
        all_acc_stds = [];
        for s = 1:length(windowed_data)
            % Calculate STD for all windows to build histogram
            win_stds = std(windowed_data(s).acc_windows, 0, 2); 
            all_acc_stds = [all_acc_stds; win_stds];
        end
        
        % Plot Histogram
        hFig = figure('Name', 'Motion Threshold Selector', 'NumberTitle', 'off');
        histogram(all_acc_stds, 50);
        title(['ACC Std Distribution for ' windowed_files(file_idx).name]);
        xlabel('ACC Standard Deviation'); ylabel('Count'); grid on;
        xline(mean(all_acc_stds), 'r--', 'Mean');
        
        % INTERACTIVE STEP: Ask user for threshold
        fprintf('Inspect the histogram.\n');
        user_input = input('Enter a motion threshold value (or press Enter for default 100): ');
        if isempty(user_input)
            motion_threshold = 100;
        else
            motion_threshold = user_input;
        end
        close(hFig); % Close histogram to clean up
        
        % 2. Apply Labeling Logic
        for s = 1:length(windowed_data)
            ppg_windows = windowed_data(s).ppg_windows;
            acc_windows = windowed_data(s).acc_windows;
            
            num_windows = size(ppg_windows, 1);
            labels = zeros(num_windows, 1); 
            
            for w = 1:num_windows
                ppg_win = ppg_windows(w, :);
                acc_win = acc_windows(w, :);
                
                % -- METRICS --
                val_acc_std = std(acc_win);
                val_ppg_std = std(ppg_win);
                
                % Flatten vectors to ensure correlation works even if dimensions vary
                val_corr = corr(ppg_win(:), acc_win(:)); 
                
                % -- LOGIC --
                is_moving    = val_acc_std > motion_threshold;
                is_saturated = val_ppg_std > SATURATION_THRESH;
                is_flatline  = val_ppg_std < FLATLINE_THRESH; % NEW CHECK
                is_mimic     = abs(val_corr) > CORR_THRESH;
                
                if is_moving || is_saturated || is_flatline || is_mimic
                    labels(w) = 1; % Artifact
                else
                    labels(w) = 0; % Clean
                end
            end
            
            % Store data
            labeled_data(s).ppg_windows = ppg_windows;
            labeled_data(s).acc_windows = acc_windows;
            labeled_data(s).labels = labels;
            labeled_data(s).win_times = windowed_data(s).win_times;
            
            % Report stats
            fprintf('  Subject %d: %d Clean | %d Artifact (%.1f%% data loss)\n', ...
                s, sum(labels==0), sum(labels==1), (sum(labels==1)/num_windows)*100);
                
            % 3. Quality Control Plot (Random Sample of Artifacts)
            % Instead of pausing every loop, we show ONE summary plot per subject
            artifact_indices = find(labels == 1);
            if ~isempty(artifact_indices)
                % Pick up to 2 random artifacts to visualize
                n_plot = min(2, length(artifact_indices));
                idx_to_plot = artifact_indices(randperm(length(artifact_indices), n_plot));
                
                figure('Name', sprintf('Subj %d Artifact Sample', s), 'Visible', 'off'); % Hidden by default
                for p = 1:n_plot
                    idx = idx_to_plot(p);
                    subplot(n_plot, 2, p*2-1); 
                    plot(ppg_windows(idx,:)); title(['Artifact PPG (Win ' num2str(idx) ')']);
                    subplot(n_plot, 2, p*2); 
                    plot(acc_windows(idx,:)); title(['ACC (std=' num2str(std(acc_windows(idx,:)), '%.1f') ')']);
                end
                % If you really want to see them, verify manually later. 
                % Uncomment 'Visible', 'on' above if you want to see them pop up.
            end
        end
        
        % Save relabeled data
        [~, name, ~] = fileparts(windowed_files(file_idx).name);
        name = erase(name, "_windowed");
        output_file = fullfile('labeled_data', sprintf('%s_labeled.mat', name));
        
        if ~exist('labeled_data', 'dir')
            mkdir('labeled_data');
        end
        
        save(output_file, 'labeled_data');
        fprintf('Saved: %s\n', output_file);
    end
    fprintf('\n=== RELABELING COMPLETE ===\n');
end