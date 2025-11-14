%% === AUTO-GENERATE PLOTS FOR ALL SUBJECTS (WORKING ACTIVITY) ===
% Outputs Working_Subject1.png ... Working_Subject15.png

clear; clc; close all;

% Path to your data folder
data_dir = 'C:\Users\User\Desktop\DSP MAMAW\DSP-MAMAW\data';

% Working activity file
act_file = 'ppg_dalia_working_data.mat';

% Load dataset
S = load(fullfile(data_dir, act_file));
data = S.data;

num_subjects = numel(data);   % usually 15
duration_sec = 60;            % seconds displayed

fprintf("Detected %d subjects.\n", num_subjects);

% Create plots folder
save_dir = fullfile(data_dir, 'plots');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Loop through all subjects
for subject_idx = 1:num_subjects
    
    fprintf("Processing Subject %d...\n", subject_idx);

    rec = data(subject_idx);

    % Extract PPG
    ppg = rec.ppg.v;
    fs_ppg = rec.ppg.fs;

    % Extract ACC
    acc = rec.acc_ppg_site.v;
    fs_acc = rec.acc_ppg_site.fs;

    % Limit to first 60 seconds
    N_ppg = min(numel(ppg), round(duration_sec * fs_ppg));
    N_acc = min(numel(acc), round(duration_sec * fs_acc));

    ppg = ppg(1:N_ppg);
    acc = acc(1:N_acc);

    t_ppg = (0:N_ppg-1) / fs_ppg;
    t_acc = (0:N_acc-1) / fs_acc;

    % Create figure
    figure('Visible','off', 'Position',[200 100 900 600]);  % <-- Invisible, faster

    % PPG
    subplot(2,1,1);
    plot(t_ppg, ppg, 'b'); grid on;
    xlabel('Time (s)');
    ylabel('PPG amplitude');
    title(sprintf('PPG Signal - Subject %d', subject_idx));

    % ACC
    subplot(2,1,2);
    plot(t_acc, acc, 'r'); grid on;
    xlabel('Time (s)');
    ylabel('ACC magnitude');
    title(sprintf('ACC Magnitude - Subject %d', subject_idx));

    sgtitle(sprintf('Working Activity - Subject %d', subject_idx));

    % Save figure
    save_name = fullfile(save_dir, sprintf('Working_Subject%d.png', subject_idx));
    saveas(gcf, save_name);

    close(gcf);  % free memory
end

fprintf("\nAll subject plots saved successfully!\n");
fprintf("Location: %s\n", save_dir);
