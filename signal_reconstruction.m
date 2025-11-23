clear; clc; close all;

% === Step 1: Load the Exported Model ===
loaded_model = load('best_model_random_forest.mat');  % Path to your saved model
best_model = loaded_model.best_model;

% === Step 2: Load Raw PPG-DaLiA Data (or your raw activity data) ===
% Load the raw data
load('data/ppg_dalia_sitting_data.mat');  % Replace with your actual raw data path

% === Step 3: Loop Through Each Subject and Process PPG/ACC ===
for s = 1:length(data)
    fprintf('Processing Subject %d of %d...\n', s, length(data));
    
    % Access the PPG and ACC signals for the current subject
    ppg = data(s).ppg.v;  % Raw PPG signal for subject s
    acc = data(s).acc_ppg_site.v;  % Raw ACC signal for subject s

    % Sampling rate for PPG and ACC (assuming it's provided in the raw data)
    fs = 64;  % Replace with the actual sampling rate of your data

    % === Step 4: Segment the Raw Data into Windows ===
    % Segment data into 8-second windows with a 2-second overlap (adjust as needed)
    window_size = 8;  % 8 seconds window length
    step_size = 2;    % 2 seconds step size for overlapping windows

    % Segment PPG and ACC signals into windows
    [ppg_windows, t_ppg] = segment_windows(ppg, fs, window_size, step_size);
    [acc_windows, ~] = segment_windows(acc, fs, window_size, step_size);

    % Check if the number of windows in both PPG and ACC are the same
    fprintf('Subject %d: PPG windows: %d, ACC windows: %d\n', s, size(ppg_windows, 1), size(acc_windows, 1));

    % === Step 5: Detect Motion Artifacts using the Trained Model ===
    num_windows = size(ppg_windows, 1);
    labels = zeros(num_windows, 1);  % Initialize labels (0 = clean, 1 = artifact)

    for w = 1:num_windows
        ppg_win = ppg_windows(w, :);
        acc_win = acc_windows(w, :);
        
        % Extract the same features that were used to train the Random Forest model
        features = extract_ppg_features(ppg_win, acc_win, fs);  % Modify as per your feature extraction
        
        % Ensure the features are in the correct format (1 x 16 vector)
        features = features(:)';  % Reshape the features to be a 1 x 16 row vector
        
        % Check the dimensions of features before prediction
        disp(['Features size: ', num2str(size(features))]);  % Display feature size

        % Ensure the features match the number of model's input features
        if size(features, 2) ~= 16  % Make sure you have 16 features (adjust if needed)
            error('Feature dimensions mismatch: Expected 16 features, but got %d', size(features, 2));
        end
        
        % Make prediction using the trained model (Random Forest)
        prediction_cell = predict(best_model, features);  % Get the prediction (cell array)

        % Convert the cell array prediction to numeric value
        prediction = cell2mat(prediction_cell);  % Convert to numeric format
        
        % Set label based on the prediction (1 = artifact, 0 = clean)
        if prediction == 1
            labels(w) = 1;  % Artifact
        else
            labels(w) = 0;  % Clean
        end
    end

    % Display number of artifact and clean windows
    fprintf('Total windows: %d\n', num_windows);
    fprintf('Artifact windows: %d\n', sum(labels == 1));
    fprintf('Clean windows: %d\n', sum(labels == 0));

    % === Step 6: Remove or Replace Motion Artifacts ===
    % Option 1: Remove Artifact Windows (keep only clean windows)
    clean_ppg = ppg_windows(labels == 0, :);
    clean_acc = acc_windows(labels == 0, :);

    % Option 2: Replace Artifact Windows (if needed)
    for w = 1:num_windows
        if labels(w) == 1  % If it's an artifact window
            % Find surrounding clean windows (just as an example, this can be adjusted)
            surrounding_clean_ppg = ppg_windows(max(1, w-5):min(num_windows, w+5), :);
            surrounding_clean_acc = acc_windows(max(1, w-5):min(num_windows, w+5), :);
            
            % Replace the artifact window with the mean of surrounding clean windows
            ppg_windows(w, :) = mean(surrounding_clean_ppg, 1);
            acc_windows(w, :) = mean(surrounding_clean_acc, 1);
        end
    end

    % === Step 7: Reconstruct the Clean Signal ===
    % After removing or replacing artifact windows, reconstruct the signal
    reconstructed_ppg = reshape(ppg_windows', [], 1);  % Reshape to get a single signal
    reconstructed_acc = reshape(acc_windows', [], 1);  % Reshape ACC signal similarly

    % === Step 8: Visualize the Results ===
    figure;
    subplot(3,1,1);
    plot(reconstructed_ppg);
    title('Reconstructed PPG Signal');
    xlabel('Samples');
    ylabel('Amplitude');

    subplot(3,1,2);
    plot(reconstructed_acc);
    title('Reconstructed ACC Signal');
    xlabel('Samples');
    ylabel('Magnitude');

    subplot(3,1,3);
    % Compare the original and reconstructed PPG signals (before and after artifact removal)
    plot(reconstructed_ppg);
    hold on;
    plot(ppg_windows(:), 'r--');  % Original PPG signal for comparison
    title('Reconstructed PPG vs Original (Red dashed)');
    xlabel('Samples');
    ylabel('Amplitude');
    legend('Reconstructed', 'Original');

    % === Step 9: Save the Reconstructed Data (Optional) ===
    save('reconstructed_data.mat', 'reconstructed_ppg', 'reconstructed_acc');
    fprintf('Reconstructed data saved to: reconstructed_data.mat\n');
end
