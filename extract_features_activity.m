file = 'labeled_data/ppg_dalia_table_soccer_data_labeled.mat';

A = load(file);
labeled_data = A.labeled_data;

feature_data = struct;

for s = 1:length(labeled_data)
    ppg_windows = labeled_data(s).ppg_windows;
    acc_windows = labeled_data(s).acc_windows;
    labels      = labeled_data(s).labels;
    fs = 64; 

    numWins = size(ppg_windows, 1);
    feat_matrix = zeros(numWins, 16); 

    for w = 1:numWins
        ppg_win = ppg_windows(w,:);
        acc_win = acc_windows(w,:);

        feat_matrix(w,:) = extract_ppg_features(ppg_win, acc_win, fs);
    end

    feature_data(s).features = feat_matrix;
    feature_data(s).labels   = labels;
end

[~, name, ~] = fileparts(file);
name = erase(name, "_labeled");
out_file = fullfile('features', sprintf('%s_features.mat', name));

if ~exist('features', 'dir')
    mkdir('features');
end

save(out_file, 'feature_data');
fprintf('\nSaved feature dataset to:\n   %s\n', out_file);