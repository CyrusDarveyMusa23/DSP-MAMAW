function feats = extract_ppg_features(ppg_win, acc_win, fs)
    ppg = double(ppg_win(:));
    acc = double(acc_win(:));

    mean_ppg = mean(ppg);
    std_ppg = std(ppg);
    skew_ppg = skewness(ppg);
    kurt_ppg = kurtosis(ppg);
    range_ppg = max(ppg) - min(ppg);
    rms_ppg = rms(ppg);
    diff_ppg = diff(ppg);
    slope_mean = mean(abs(diff_ppg));
    slope_std = std(diff_ppg);

    N = length(ppg);
    f = (0:N-1) * (fs/N);
    P = abs(fft(ppg));

    hr_band = (f >= 0.5 & f <= 4);
    band_power = sum(P(hr_band));
    total_power = sum(P);
    power_ratio = band_power / total_power;

    std_acc = std(acc);
    rms_acc = rms(acc);
    range_acc = max(acc) - min(acc);

    [pks, ~] = findpeaks(ppg, 'MinPeakDistance', round(fs * 0.3));
    peak_count = length(pks);
    zc = sum(ppg(1:end-1).*ppg(2:end) < 0);

    feats = [
        mean_ppg
        std_ppg
        skew_ppg
        kurt_ppg
        range_ppg
        rms_ppg
        slope_mean
        slope_std
        band_power
        total_power
        power_ratio
        std_acc
        rms_acc
        range_acc
        peak_count
        zc
    ];
end