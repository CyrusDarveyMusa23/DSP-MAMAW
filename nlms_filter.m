function [y_hat, e, W] = nlms_filter_clean(x, R, mu, L, delta)
    x = x(:);                         
    [N, M] = size(R);               
    TotalTaps = L * M;               

    w_current = zeros(TotalTaps, 1);   
    W = zeros(TotalTaps, N);          
    e = zeros(N, 1);                 
    y_hat = zeros(N, 1);               
    
    r_vec = zeros(TotalTaps, 1);       
    
    for n = 1:N
        for m = 1:M
            r_start_idx = (m-1)*L + 1;
            r_end_idx = m*L;
            
            r_vec(r_start_idx + 1 : r_end_idx) = r_vec(r_start_idx : r_end_idx - 1);
            r_vec(r_start_idx) = R(n, m);
        end

        y_hat(n) = w_current.' * r_vec;
        e(n) = x(n) - y_hat(n);
        norm_r2 = r_vec.' * r_vec;
        w_current = w_current + (mu / (delta + norm_r2)) * e(n) * r_vec;
        W(:, n) = w_current;
    end
end