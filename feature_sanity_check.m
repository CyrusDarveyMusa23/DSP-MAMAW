%% Feature Sanity Check
load('training_dataset.mat');

figure('Position', [100 100 1200 400]);

% Plot first 2 features
subplot(1,3,1);
gscatter(all_features(:,1), all_features(:,2), all_labels, 'rb', 'xo');
xlabel('Feature 1 (Mean PPG)'); ylabel('Feature 2 (Std PPG)');
title('Feature Space - First 2 Features');
legend('Clean', 'Artifact');

% Check if features are constant
subplot(1,3,2);
feature_variance = var(all_features);
bar(feature_variance);
xlabel('Feature Index'); ylabel('Variance');
title('Feature Variance');
grid on;

% Class distribution
subplot(1,3,3);
histogram(all_labels);
xlabel('Class (0=Clean, 1=Artifact)'); ylabel('Count');
title('Class Distribution');
xticks([0 1]);