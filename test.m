%% Alternative split that ensures both classes in test set
function [Xtrain, Ytrain, Xtest, Ytest] = balanced_split(features, labels, test_ratio)
    % Get indices of each class
    clean_idx = find(labels == 0);
    artifact_idx = find(labels == 1);
    
    % Calculate how many from each class for test set
    n_test_clean = round(length(clean_idx) * test_ratio);
    n_test_artifact = round(length(artifact_idx) * test_ratio);
    
    % Randomly select test samples from each class
    test_clean = randsample(clean_idx, n_test_clean);
    test_artifact = randsample(artifact_idx, n_test_artifact);
    
    % Combine test indices
    test_idx = [test_clean; test_artifact];
    train_idx = setdiff(1:length(labels), test_idx);
    
    % Create splits
    Xtrain = features(train_idx, :);
    Ytrain = labels(train_idx);
    Xtest = features(test_idx, :);
    Ytest = labels(test_idx);
end