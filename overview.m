function ppg_signal_viewer()

    activity_files = dir('data/*_data.mat');  
    if isempty(activity_files)
        error('No activity .mat files found in data/');
    end

    activity_names = {activity_files.name};

    fig = uifigure('Name','PPG-DaLiA Activity Viewer','Position',[300 150 500 220]);

    activityDD = uidropdown(fig, ...
        'Items', activity_names, ...
        'Position',[40 150 420 30], ...
        'ValueChangedFcn', @(src,evt) updateSubjectDropdown(src.Value));

    subjectDD = uidropdown(fig, ...
        'Items', {'Select subject'}, ...
        'Position',[40 100 420 30]);

    uibutton(fig,'Text','VIEW SIGNALS','Position',[170 40 160 40], ...
        'ButtonPushedFcn', @(src,evt) visualizeSignals());

    % --- update subject dropdown
    function updateSubjectDropdown(selectedFile)
        A = load(fullfile('data', selectedFile));
        data = A.data;
        if iscell(data), data = data{1}; end

        numSubjects = length(data);
        subjectDD.Items = arrayfun(@(x)sprintf('Subject %d', x), 1:numSubjects, 'UniformOutput',false);
        subjectDD.Value = subjectDD.Items{1};
    end

    % --- view signals
    function visualizeSignals()
        activity_file = fullfile('data', activityDD.Value);
        subject_str   = subjectDD.Value;
        subject_num   = sscanf(subject_str, 'Subject %d');
        visualize_activity_subject(activity_file, subject_num);
    end

end

%% ===================================================================
%%   MAIN VISUALIZATION (Now includes RF-based reconstruction)
%% ===================================================================
function visualize_activity_subject(activity_mat_file, subject_num)

    % Load RF model
    model_file = 'best_model_random_forest.mat'; % Or 'Best_Model_Final.mat'
    if ~isfile(model_file)
        uialert(uifigure, 'Model file not found!', 'Error');
        return;
    end
    M = load(model_file);
    
    % Handle different saving formats from previous steps
    if isfield(M, 'best_model'), mdl = M.best_model;
    elseif isfield(M, 'final_rf_model'), mdl = M.final_rf_model;
    elseif isfield(M, 'final_model_struct'), mdl = M.final_model_struct.model;
    else, error('Could not find model variable in .mat file'); end

    % 2. Load Data
    A = load(activity_mat_file);
    data = A.data;
    if iscell(data), data = data{1}; end
    D = data(subject_num);

    % 3. Extract Signals
    raw_ppg = D.ppg.v(:);
    fs_ppg  = D.ppg.fs;

    raw_acc = D.acc_ppg_site.v(:);
    fs_acc  = D.acc_ppg_site.fs;

    % Load Filtered Data (If available)
    [~, base, ~] = fileparts(activity_mat_file);
    filtered_file = fullfile('filtered_data', base + "_filtered.mat");
    filt_ppg = [];
    if isfile(filtered_file)
        F = load(filtered_file);
        filtered_data = F.filtered_data;
        filt_ppg = filtered_data(subject_num).ppg_filt(:);
    end

    % Resample ACC to PPG
    if fs_ppg ~= fs_acc
        [p,q] = rat(fs_ppg/fs_acc);
        raw_acc = resample(raw_acc, p, q);
    end

    % Match lengths
    N = min(length(raw_ppg), length(raw_acc));
    raw_ppg = raw_ppg(1:N);
    raw_acc = raw_acc(1:N);
    if ~isempty(filt_ppg)
        filt_ppg = filt_ppg(1:N);
    end

    %% === Step: Windowing for RF model ===
    window_size = 8;    % seconds
    step_size   = 2;    % seconds

    [ppg_windows, ~] = segment_windows(raw_ppg, fs_ppg, window_size, step_size);
    [acc_windows, ~] = segment_windows(raw_acc, fs_ppg, window_size, step_size);

    num_windows = size(ppg_windows,1);
    labels = zeros(num_windows,1);

    %% === Step: RF-based artifact detection + reconstruction ===
    reconstructed_windows = ppg_windows;  % initialize

    hWait = waitbar(0, 'Running Artifact Detection and Reconstruction...');
    for w = 1:num_windows

        ppg_win = ppg_windows(w,:);
        acc_win = acc_windows(w,:);

        feats = extract_ppg_features(ppg_win, acc_win, fs_ppg);
        feats = feats(:)';  % ensure 1x16

        % Predict using the trained RF model
        pred_cell = predict(mdl, feats);
        pred = str2double(pred_cell);

        labels(w) = pred;

        % If artifact → replace with mean of neighbors
        if pred == 1
            lo = max(1, w-3);
            hi = min(num_windows, w+3);

            neighbors = ppg_windows(lo:hi, :);
            reconstructed_windows(w,:) = mean(neighbors,1);
        end

        % Update the waitbar
        waitbar(w/num_windows, hWait, sprintf('Processing Window %d of %d...', w, num_windows));
    end
    close(hWait);

    reconstructed_ppg = reshape(reconstructed_windows', [], 1);

    %% === Trim reconstructed to match original ===
    reconstructed_ppg = reconstructed_ppg(1:length(raw_ppg));

    %% === Plot Results ===
    t = (0:length(raw_ppg)-1)/fs_ppg;
    idx = t <= 10;

    fig = figure('Name', sprintf('Subject %d Analysis', subject_num), ...
        'Position',[100 100 1000 800], 'Color', 'w');

    subplot(4,1,1);
    plot(t(idx), raw_ppg(idx),'r'); grid on;
    title('Raw PPG (First 10 seconds)'); ylabel('Amplitude');

    subplot(4,1,2);
    plot(t(idx), raw_acc(idx),'k'); grid on;
    title('Raw ACC'); ylabel('|ACC|');

    subplot(4,1,3); 
    if isempty(filt_ppg)
        text(0.2,0.5,'NO FILTERED PPG FOUND','FontSize',16);
    else
        plot(t(idx), filt_ppg(idx),'b'); grid on;
        title('Filtered PPG (NLMS)'); ylabel('Amplitude');
    end

    subplot(4,1,4);
    plot(t(idx), reconstructed_ppg(idx),'g'); grid on;
    title('Reconstructed PPG (Random Forest Artifact Reduction)');
    xlabel('Time (s)'); ylabel('Amplitude');

    fprintf("✔ Reconstruction complete. RF detected %d artifact windows.\n", sum(labels==1));

end
