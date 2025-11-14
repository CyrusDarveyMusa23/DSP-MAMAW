%% === VISUALIZE ONE SUBJECT FOR WALKING ACTIVITY (PPG + ACC) ===
% Shows only one subject per run, but auto-saves the figure with
% filename: <activity>_Subject<idx>.png

clear; clc; close all;

% Path to your "data" folder
data_dir = 'C:\Users\User\Desktop\DSP MAMAW\DSP-MAMAW\data';

% Walking activity file
act_file = 'ppg_dalia_walking_data.mat';

% Choose which subject to visualize
subject_idx = 1;      % <-- Change this to 1,2,...,15

% Number of seconds to visualize
duration_sec = 60;

% Load dataset
S = load(fullfile(data_dir, act_file));
data = S.data;

% Safety check
if subject_idx > numel(data)
    error('Subject %d does not exist. This file has only %d subjects.', ...
           subject_idx, numel(data));
end

rec = data(subject_idx);

% Extract PPG
ppg = rec.ppg.v;
fs_ppg = rec.ppg.fs;

% Extract ACC magnitude
acc = rec.acc_ppg_site.v;
fs_acc = rec.acc_ppg_site.fs;

% Limit to first N seconds
N_ppg = min(numel(ppg), round(duration_sec * fs_ppg));
N_acc = min(numel(acc), round(duration_sec * fs_acc));

ppg = ppg(1:N_ppg);
acc = acc(1:N_acc);

t_ppg = (0:N_ppg-1) / fs_ppg;
t_acc = (0:N_acc-1) / fs_acc;

% Figure
figure('Name', sprintf('Walking - Subject %d', subject_idx), ...
       'NumberTitle','off','Position',[200 100 900 600]);

% --- Plot PPG ---
subplot(2,1,1);
plot(t_ppg, ppg, 'b');
grid on;
xlabel('Time (s)');
ylabel('PPG amplitude');
title(sprintf('PPG Signal - Subject %d', subject_idx));

% --- Plot ACC magnitude ---
subplot(2,1,2);
plot(t_acc, acc, 'r');
grid on;
xlabel('Time (s)');
ylabel('ACC magnitude');
title(sprintf('ACC Magnitude - Subject %d', subject_idx));

sgtitle(sprintf('Walking Activity - Subject %d', subject_idx));

% Auto-save figure
save_dir = fullfile(data_dir, 'plots');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

activity_name = "Walking";
save_name = fullfile(save_dir, sprintf('%s_Subject%d.png', activity_name, subject_idx));
saveas(gcf, save_name);

fprintf('\nSaved plot as:\n%s\n', save_name);
