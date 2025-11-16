% Check one of your labeled files
load('labeled_data/ppg_dalia_sitting_data_labeled.mat');
fprintf('Labels in sitting data: %d clean, %d artifact\n', ...
    sum(labeled_data(1).labels==0), sum(labeled_data(1).labels==1));