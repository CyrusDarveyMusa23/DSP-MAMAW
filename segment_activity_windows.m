file = 'filtered_data/ppg_dalia_working_data_filtered.mat';
win_len_sec = 8;
step_sec = 2;

A = load(file);
filtered_data = A.filtered_data;

out_folder = fullfile(pwd, 'windowed_data');
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
    fprintf('Created folder: %s\n', out_folder);
end

windowed_data = struct;

for s = 1:length(filtered_data)
    D = filtered_data(s);

    ppg = D.ppg_filt(:);
    acc = D.acc_mag_rs(:);
    fs = D.ppg.fs;

    [ppg_win, t_ppg] = segment_windows(ppg, fs, win_len_sec, step_sec);
    [acc_win, ~] = segment_windows(acc, fs, win_len_sec, step_sec);

    windowed_data(s).ppg_windows = ppg_win;
    windowed_data(s).acc_windows = acc_win;
    windowed_data(s).win_times = t_ppg;
end

[~, name, ~] = fileparts(file);
name = erase(name, "_filtered");
out_file = fullfile(out_folder, sprintf('%s_windowed.mat', name));
save(out_file, 'windowed_data');
fprintf('\nSaved windowed dataset to:\n   %s\n', out_file);
