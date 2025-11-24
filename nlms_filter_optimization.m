function run_nlms_ppg_data(activity_mat_file)
    % 1. Load Data
    A = load(activity_mat_file);
    data = A.data;      
    fprintf('Loaded %d subjects from %s\n', length(data), activity_mat_file);
    
    % 2. Setup Constants
    L     = 64;       % Filter length (approx 2 sec at 32Hz)
    delta = 1e-3;     % Regularization
    
    % List of step sizes to test per subject
    mu_candidates = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25];
  
    filtered_data = data;   
    
    for s = 1:length(data)
        fprintf('------------------------------------------------\n');
        fprintf('Processing subject %d of %d...\n', s, length(data));
        D = data(s);     
        
        % --- A. PRE-PROCESSING ---
        ppg     = double(D.ppg.v(:));
        fs_ppg  = D.ppg.fs;
        
        % Use all 3 accelerometer channels (Nx3) instead of flattening them
        acc_raw = double(D.acc_ppg_site.v); 
        fs_acc  = D.acc_ppg_site.fs;

        if size(acc_raw, 1) < size(acc_raw, 2)
            acc_raw = acc_raw';
        end

        [~, num_channels] = size(acc_raw);
  
        % Resample ACC to match PPG (usually 32Hz in Dalia)
        if fs_acc ~= fs_ppg
            [p, q] = rat(fs_ppg / fs_acc);
            % Resample each channel (column) independently
            new_len = ceil(size(acc_raw, 1) * p/q);
            R = zeros(new_len, num_channels);
            for ch = 1:3
                R(:, ch) = resample(acc_raw(:, ch), p, q);
            end
        else
            R = acc_raw;
        end
        
        % Ensure lengths match exactly
        N = min(length(ppg), size(R, 1));
        ppg = ppg(1:N);
        R   = R(1:N, :);
        
        % --- B. OPTIMIZATION LOOP (Find Best Mu) ---
        best_score = -inf;
        best_mu = 0.01; % Default fallback
        
        % We only test on the first 1000 samples (approx 30s) to save time
        N_test = min(N, 1000); 
        
        for k = 1:length(mu_candidates)
            test_mu = mu_candidates(k);
            
            % Run filter on snippet
            [~, e_test, ~] = nlms_filter(ppg(1:N_test), R(1:N_test, :), test_mu, L, delta);
            
            % Calculate Quality Score (Spectral Peakedness in HR band)
            score = calculate_quality_score(e_test, fs_ppg);
            
            if score > best_score
                best_score = score;
                best_mu = test_mu;
            end
        end
        
        fprintf('  > Optimal mu found: %.4f (Score: %.2f)\n', best_mu, best_score);

        % --- C. FINAL FILTERING ---
        % Run on FULL dataset using the best mu found
        [y_hat, ppg_filt, W] = nlms_filter(ppg, R, best_mu, L, delta);
        
        % Store results
        filtered_data(s).ppg_filt = ppg_filt;
        filtered_data(s).acc_rs = R; % Store resampled acc
        filtered_data(s).best_mu = best_mu;
    end

    % 3. Save Results
    out_folder = fullfile(pwd, 'filtered_data');
    if ~exist(out_folder, 'dir')
        mkdir(out_folder);
    end
    [~, base_name, ~] = fileparts(activity_mat_file);
    out_file = fullfile(out_folder, sprintf('%s_filtered.mat', base_name));
    save(out_file, 'filtered_data');
    fprintf('\nSaved filtered data to: %s\n', out_file);
    
    % 4. Plotting (First Subject Only to avoid popup spam)
    plot_results(data(1), filtered_data(1), 1, 10);
end

%% --- HELPER FUNCTIONS ---

function score = calculate_quality_score(signal, fs)
    % Calculates how "peaky" the signal is in the Heart Rate band (0.5-3.0Hz)
    N_fft = 1024;
    if length(signal) < N_fft
        N_fft = length(signal);
    end
    
    [pxx, f] = pwelch(signal, window(@hamming, min(256, length(signal))), [], N_fft, fs);
    
    hr_band = f >= 0.8 & f <= 3.0; % Strict HR band (48 - 180 BPM)
    
    peak_power = max(pxx(hr_band));
    mean_noise = mean(pxx(~hr_band)) + eps;
    
    score = peak_power / mean_noise;
end

function [y_hat, e, W] = nlms_filter(x, R, mu, L, delta)
    % Optimized NLMS Filter
    x = x(:);
    [N, M] = size(R); % M is number of accel channels (3)
    TotalTaps = L * M;
    
    w = zeros(TotalTaps, 1);
    W = zeros(TotalTaps, N);
    e = zeros(N, 1);
    y_hat = zeros(N, 1);
    
    buffer = zeros(L, M); 
    
    for n = 1:N
        % Shift buffer and add new sample
        buffer(2:end, :) = buffer(1:end-1, :);
        buffer(1, :) = R(n, :);
        
        r_vec = buffer(:); 
        
        % Filter
        y_hat(n) = w' * r_vec;
        e(n) = x(n) - y_hat(n);
        
        % Update
        norm_r2 = r_vec' * r_vec;
        w = w + (mu / (delta + norm_r2)) * e(n) * r_vec;
        W(:, n) = w;
    end
end

function plot_results(orig_struct, filt_struct, subj_idx, dur_sec)
    ppg_raw  = double(orig_struct.ppg.v(:));
    ppg_clean = filt_struct.ppg_filt(:);
    acc_ref   = filt_struct.acc_rs(:, 1); % Show X-axis as reference
    fs = orig_struct.ppg.fs;
    
    N = min([length(ppg_raw), length(ppg_clean), length(acc_ref)]);
    t = (0:N-1)/fs;
    
    idx = t <= dur_sec;
    
    figure('Color','w', 'Position', [100 100 1000 800]);
    
    subplot(3,1,1);
    plot(t(idx), ppg_raw(idx), 'r'); title('Raw PPG (Noisy)'); grid on;
    xlim([0 dur_sec]);
    
    subplot(3,1,2);
    plot(t(idx), acc_ref(idx), 'k'); title('Motion Reference (Acc-X)'); grid on;
    xlim([0 dur_sec]);
    
    subplot(3,1,3);
    plot(t(idx), ppg_clean(idx), 'b', 'LineWidth', 1.2); 
    title(['Cleaned PPG | Best \mu = ' num2str(filt_struct.best_mu)]); 
    xlabel('Time (s)'); grid on;
    xlim([0 dur_sec]);
end