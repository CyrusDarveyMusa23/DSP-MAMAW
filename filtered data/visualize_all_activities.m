function visualize_all_activities(data_dir, subject_idx, duration_sec)
% AUTO-DETECT & VISUALIZE ALL PPG-DALIA .MAT FILES
%
%   visualize_all_activities(data_dir, subject_idx, duration_sec)
%
%   data_dir     : folder containing .mat files
%   subject_idx  : subject to visualize (default = 1)
%   duration_sec : seconds to plot (default = 60)

if nargin < 2, subject_idx = 1; end
if nargin < 3, duration_sec = 60; end

% === FIND ALL MAT FILES IN THE FOLDER ===
files = dir(fullfile(data_dir, '*.mat'));

if isempty(files)
    error('No .mat files found in: %s', data_dir);
end

figure('Name','PPG Across Activities','NumberTitle','off');

plot_idx = 0;

for k = 1:numel(files)

    filename = fullfile(data_dir, files(k).name);

    % Try loading the file safely
    try
        S = load(filename);
    catch
        fprintf('Skipping file (cannot load): %s\n', files(k).name);
        continue;
    end

    % Check if it contains "data"
    if ~isfield(S, 'data')
        fprintf('Skipping file (no ''data'' variable): %s\n', files(k).name);
        continue;
    end

    data = S.data;

    if subject_idx > numel(data)
        fprintf('Skipping %s (subject %d not available)\n', files(k).name, subject_idx);
        continue;
    end

    rec = data(subject_idx);

    % Check if PPG exists
    if ~isfield(rec, 'ppg') || ~isfield(rec.ppg, 'v')
        fprintf('Skipping %s (no PPG field)\n', files(k).name);
        continue;
    end

    % Extract PPG
    ppg = rec.ppg.v;
    fs  = rec.ppg.fs;
    N   = min(numel(ppg), round(duration_sec * fs));
    ppg = ppg(1:N);
    t   = (0:N-1) / fs;

    % Plot
    plot_idx = plot_idx + 1;
    subplot(numel(files), 1, plot_idx);

    plot(t, ppg);
    grid on;

    title(strrep(files(k).name, '_', '\_'), 'Interpreter','none');
    xlabel('Time (s)');
    ylabel('PPG');
end

end
