function visualize_activity_subject(activity_mat_file, subject_num)
% Visualize the first 10 seconds of raw PPG and ACC for a given subject
% inside an activity .mat file

%% === Load activity file ===
A = load(activity_mat_file);

if ~isfield(A, 'data')
    error('The file %s does not contain a ''data'' variable.', activity_mat_file);
end

data = A.data;

% Unwrap cell if needed
if iscell(data)
    data = data{1};
end

%% === Validate subject number ===
if subject_num > length(data)
    error('Subject %d does not exist in this activity file. Max is %d.', ...
           subject_num, length(data));
end

D = data(subject_num);

fprintf('Loaded %s | Subject %d\n', activity_mat_file, subject_num);

%% === Extract PPG ===
ppg = D.ppg.v(:);
fs_ppg = D.ppg.fs;
t_ppg = (0:length(ppg)-1)/fs_ppg;

%% === Extract ACC magnitude ===
acc = D.acc_ppg_site.v(:);
fs_acc = D.acc_ppg_site.fs;

% Resample ACC to match PPG sampling rate
if fs_ppg ~= fs_acc
    [p, q] = rat(fs_ppg/fs_acc);
    acc = resample(acc, p, q);
end

% Match lengths
N = min(length(ppg), length(acc));
ppg = ppg(1:N);
acc = acc(1:N);
t = (0:N-1)/fs_ppg;

%% === Limit to first 10 seconds ===
duration = 10;
idx = t <= duration;

%% === Plot Results ===
figure('Name', sprintf('Activity: %s - Subject %d (First 10s)', ...
    activity_mat_file, subject_num), ...
    'Position', [200 200 1000 600]);

% --- RAW PPG ---
subplot(2,1,1);
plot(t(idx), ppg(idx), 'r');
title(sprintf('Raw PPG - Subject %d (First 10 seconds)', subject_num));
xlabel('Time (s)');
ylabel('PPG Amplitude');
grid on;

% --- ACC magnitude ---
subplot(2,1,2);
plot(t(idx), acc(idx), 'k');
title(sprintf('ACC Magnitude - Subject %d (First 10 seconds)', subject_num));
xlabel('Time (s)');
ylabel('|ACC| (mG)');
grid on;

end
