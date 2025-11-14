%% === AUTO-GENERATE CONTINUOUS PPG + ACC PLOTS FOR ALL SUBJECTS ===
% Outputs AllActivities_Subject1.png ... AllActivities_Subject15.png
% Based on ppg_dalia_all_activities_data.mat (NO activity labels)

clear; clc; close all;

% Path to your data folder
data_dir = 'C:\Users\User\Desktop\DSP MAMAW\DSP-MAMAW\data';

% Combined all-activities file (continuous signals)
act_file = 'ppg_dalia_all_activities_data.mat';

% Load dataset
S = load(fullfile(data_dir, act_file));
data = S.data;

num_subjects = numel(data);
duration_sec = 60;  % display first 60 seconds only

fprintf("Detected %d subjects in combined ALL ACTIVITIES dataset.\n", num_subjects);

% Create output folder
save_dir = fullfile(data_dir, 'plots_all_activities_continuous');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Loop through all subjects
for subject_idx = 1:num_subjects

    fprintf("Processing Subject %d...\n", subject_idx);

    rec = data(subject_idx);

    % Extract continuous PPG
    ppg = rec.ppg.v;
    fs_ppg = rec.ppg.fs;

    % Extract continuous ACC magnitude
    acc = rec.acc_ppg_site.v;
    fs_acc = rec.acc_ppg_site.fs;

    % Limit to duration (optional)
    N_ppg = min(numel(ppg), round(duration_sec * fs_ppg));
    N_acc = min(numel(acc), round(duration_sec * fs_acc));

    ppg = ppg(1:N_ppg);
    acc = acc(1:N_acc);

    t_ppg = (0:N_ppg-1) / fs_ppg;
    t_acc = (0:N_acc-1) / fs_acc;

    % === Create figure (hidden for fast processing) ===
    figure('Visible','off','Position',[200 100 900 600]);

    % --- PPG Plot ---
    subplot(2,1,1);
    plot(t_ppg, ppg, 'b'); grid on;
    xlabel('Time (s)');
    ylabel('PPG amplitude');
    title(sprintf('Continuous PPG - Subject %d', subject_idx));

    % --- ACC Plot ---
    subplot(2,1,2);
    plot(t_acc, acc, 'r'); grid on;
    xlabel('Time (s)');
    ylabel('ACC magnitude');
    title(sprintf('Continuous ACC - Subject %d', subject_idx));

    sgtitle(sprintf('Continuous Signal - All Activities - Subject %d', subject_idx));

    % === Save output ===
    save_name = fullfile(save_dir, sprintf('AllActivities_Subject%d.png', subject_idx));
    saveas(gcf, save_name);

    close(gcf);  % free memory
end

fprintf("\nAll continuous plots saved successfully!\n");
fprintf("Location: %s\n", save_dir);
