%% extract_ppg_features.m - Feature extraction function
function features = extract_ppg_features(ppg, acc, fs)
% Extract 16 features from PPG and ACC windows
    
    % Ensure row vectors
    ppg = ppg(:)';
    acc = acc(:)';
    
    features = zeros(1, 16);
    
    % 1. Statistical Features (PPG)
    features(1) = mean(ppg);
    features(2) = std(ppg);
    features(3) = skewness(ppg);
    features(4) = kurtosis(ppg);
    
    % 2. Frequency Domain Features (PPG)
    N = length(ppg);
    if N > 0
        f = (0:N-1)*(fs/N);
        Pxx = abs(fft(ppg - mean(ppg))).^2 / N;
        
        % Find dominant frequency in reasonable HR range (0.67-4 Hz ≈ 40-240 BPM)
        valid_idx = (f >= 0.67) & (f <= 4.0) & (f > 0);
        if any(valid_idx)
            [~, max_idx] = max(Pxx(valid_idx));
            valid_freqs = f(valid_idx);
            features(5) = valid_freqs(max_idx); % Dominant frequency
            
            % Spectral entropy
            Pxx_norm = Pxx(valid_idx) / sum(Pxx(valid_idx));
            features(6) = -sum(Pxx_norm .* log2(Pxx_norm + eps));
        else
            features(5) = 0;
            features(6) = 0;
        end
    else
        features(5) = 0;
        features(6) = 0;
    end
    
    % 3. Motion-related Features (ACC)
    features(7) = mean(acc); % Average motion
    features(8) = std(acc);  % Motion variability
    features(9) = max(acc);  % Peak motion
    
    % 4. PPG-ACC Correlation
    if length(ppg) == length(acc) && length(ppg) > 1
        valid_corr = corr(ppg(:), acc(:));
        if isnan(valid_corr)
            features(10) = 0;
        else
            features(10) = valid_corr;
        end
    else
        features(10) = 0;
    end
    
    % 5. Signal Quality Index (SQI) - variance ratio
    ppg_var = var(ppg);
    acc_var = var(acc);
    if acc_var > 0
        features(11) = ppg_var / acc_var;
    else
        features(11) = ppg_var;
    end
    
    % 6. Additional temporal features
    features(12) = rms(ppg);
    features(13) = median(ppg);
    features(14) = iqr(ppg);
    
    % 7. Zero-crossing rate
    if length(ppg) > 1
        zc = sum(diff(ppg > mean(ppg)) ~= 0);
        features(15) = zc / length(ppg);
    else
        features(15) = 0;
    end
    
    % 8. Peak-to-peak amplitude
    features(16) = max(ppg) - min(ppg);
end