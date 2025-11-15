function run_nlms(activity_mat_file)
    A = load(activity_mat_file);
    data = A.data;      
    fprintf('Loaded %d subjects from %s\n', length(data), activity_mat_file);
    
    L     = 8;      
    mu    = 0.01;   
    delta = 1e-3;   
  
    filtered_data = data;   
    
    for s = 1:length(data)
        fprintf('Processing subject %d of %d...\n', s, length(data));
        D = data(s);     
        
        ppg     = double(D.ppg.v(:));
        fs_ppg  = D.ppg.fs;
        acc_mag = double(D.acc_ppg_site.v(:)); 
        fs_acc  = D.acc_ppg_site.fs;
  
        R = acc_mag(:);
        
        if fs_acc ~= fs_ppg
            [p, q] = rat(fs_ppg / fs_acc);
            R = resample(R, p, q);
        end
        
        N = min(length(ppg), length(R));
        ppg = ppg(1:N);
        R   = R(1:N);
        
        [y_hat, ppg_filt, W] = nlms_filter(ppg, R, mu, L, delta);
        
        filtered_data(s).ppg_filt = ppg_filt;
        filtered_data(s).acc_mag_rs = R; 
    end

    out_folder = fullfile(pwd, 'filtered_data');
    if ~exist(out_folder, 'dir')
        mkdir(out_folder);
        fprintf('Created folder: %s\n', out_folder);
    end

    [~, base_name, ~] = fileparts(activity_mat_file);
    out_file = fullfile(out_folder, sprintf('%s_filtered.mat', base_name));

    save(out_file, 'filtered_data');
    fprintf('\nSaved filtered data to:\n   %s\n', out_file);
    
    dur = 8;   
    fprintf('\nPlotting raw, filtered, ACC, and comparison for first %d seconds...\n', dur);
    
    for s = 1:length(data)
        D = data(s);
        ppg        = double(D.ppg.v(:));
        fs_ppg     = D.ppg.fs;
        ppg_filt   = filtered_data(s).ppg_filt(:);
        acc_mag_rs = filtered_data(s).acc_mag_rs(:); 

        N = min([length(ppg), length(ppg_filt), length(acc_mag_rs)]);
        ppg        = ppg(1:N);
        ppg_filt   = ppg_filt(1:N);
        acc_mag_rs = acc_mag_rs(1:N);

        t = (0:N-1) / fs_ppg;
        idx = t <= dur;
        
        figure('Name', sprintf('Subject %d - %s', s, activity_mat_file), ...
            'Position', [200 200 1100 800]);

        subplot(4,1,1);
        plot(t(idx), ppg(idx), 'r');
        title(sprintf('Subject %d - Raw PPG', s));
        ylabel('Amplitude');
        grid on;

        subplot(4,1,2);
        plot(t(idx), ppg_filt(idx), 'b');
        title('Filtered PPG (NLMS Clean Output)');
        ylabel('Amplitude');
        grid on;
     
        subplot(4,1,3);
        plot(t(idx), acc_mag_rs(idx), 'k');
        title('Accelerometer Magnitude (Motion Reference)');
        ylabel('|ACC|');
        grid on;
        
        subplot(4,1,4);
        plot(t(idx), ppg(idx), 'r'); hold on;
        plot(t(idx), ppg_filt(idx), 'b');
        title('Raw vs Filtered Comparison');
        xlabel('Time (s)');
        ylabel('Amplitude');
        legend('Raw', 'Filtered');
        grid on;
    end
end