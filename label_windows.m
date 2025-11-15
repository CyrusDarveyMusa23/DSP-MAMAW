file = 'windowed_data/ppg_dalia_lunch_break_data_windowed.mat';
motion_threshold = 0.03;

A = load(file);
windowed_data = A.windowed_data;

out_folder = fullfile(pwd, 'labeled_data');
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
    fprintf('Created folder: %s\n', out_folder);
end

labeled_data = struct;

for s = 1:length(windowed_data)
    acc_win = windowed_data(s).acc_windows;
    numWins = size(acc_win, 1);

    labels = zeros(numWins, 1);

    for w = 1:numWins
        motion_level = std(acc_win(w,:));

        if motion_level > motion_threshold
            labels(w) = 1;
        else
            labels(w) = 0;
        end
    end

    labeled_data(s).ppg_windows = windowed_data(s).ppg_windows;
    labeled_data(s).acc_windows = acc_win;
    labeled_data(s).win_times   = windowed_data(s).win_times;
    labeled_data(s).labels      = labels;
end

[~, name, ~] = fileparts(file);
name = erase(name, "_windowed");
out_file = fullfile(out_folder, sprintf('%s_labeled.mat', name));
save(out_file, 'labeled_data');
fprintf('\nSaved labeled dataset to:\n   %s\n', out_file);
