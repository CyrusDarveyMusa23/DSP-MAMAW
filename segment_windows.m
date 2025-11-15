function [win_segments, win_times] = segment_windows(signal, fs, win_len_sec, step_sec)
    signal = signal(:);
    N = length(signal);

    win_len = win_len_sec * fs;
    step = step_sec * fs;

    numWins = floor((N - win_len) / step) + 1;

    win_segments = zeros(numWins, win_len);
    win_times = zeros(numWins, 1);

    idx = 1;
    for w = 1:numWins
        start_i = idx;
        end_i = idx + win_len - 1;

        win_segments(w, :) = signal(start_i:end_i);
        win_times(w) = (start_i - 1) / fs;

        idx = idx + step;
    end
end