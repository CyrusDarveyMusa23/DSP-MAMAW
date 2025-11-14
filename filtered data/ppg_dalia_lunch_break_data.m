%% === AUTO-GENERATE PLOTS FOR ALL SUBJECTS (LUNCH BREAK ACTIVITY) ===
% Outputs LunchBreak_Subject1.png ... LunchBreak_Subject15.png

clear; clc; close all;

% Path to your data folder
data_dir = 'C:\Users\User\Desktop\DSP MAMAW\DSP-MAMAW\data';

% Lunch break activity file
act_file = 'ppg_dalia_lunch_break_data.mat';   % <-- UPDATED

% Load dataset
S = load(fullfile(data_dir, act_file));
data = S.data;

num_subjects = numel(data);     % usually around 15
duration_sec = 60;              % seconds to visualize

fprintf("Detected %d subjects in Lunch Break dataset.\n", num_subjects);

% Create output folder
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

    % Extract ACC magnitude
    acc = rec.acc_ppg_site.v;
    fs_acc = rec.acc_ppg_site.fs;

    % Limit to first duration_sec seconds
    N_ppg = min(numel(ppg), round(duration_sec * fs_ppg));
    N_acc = min(numel(acc), round(duration_sec * fs_acc));

    ppg = ppg(1:N_ppg);
    acc = acc(1:N_acc);

    t_ppg = (0:N_ppg-1) / fs_ppg;
    t_acc = (0:N_acc-1) / fs_acc;

    % === Create figure ===
    figure('Visible','off','Position',[200 100 900 600]);

    % --- PPG subplot ---
    subplot(2,1,1);
    plot(t_ppg, ppg, 'b'); 
    grid on;
    xlabel('Time (s)');
    ylabel('PPG amplitude');
    title(sprintf('PPG Signal - Subject %d', subject_idx));

    % --- ACC subplot ---
    subplot(2,1,2);
    plot(t_acc, acc, 'r'); 
    grid on;
    xlabel('Time (s)');
    ylabel('ACC magnitude');
    title(sprintf('ACC Magnitude - Subject %d', subject_idx));

    sgtitle(sprintf('Lunch Break Activity - Subject %d', subject_idx));

    % === Save figure ===
    save_name = fullfile(save_dir, sprintf('LunchBreak_Subject%d.png', subject_idx));
    saveas(gcf, save_name);

    close(gcf);
end

fprintf("\nAll Lunch Break subject plots saved successfully!\n");
fprintf("Location: %s\n", save_dir);
