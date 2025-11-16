%% Load all trained models
clear; clc;

% Load the complete models file
load('results/trained_models.mat');

% List all available models
model_names = fieldnames(models);
fprintf('Available trained models:\n');
for i = 1:length(model_names)
    fprintf('  %d. %s\n', i, model_names{i});
end

% Load the best model separately
load('results/best_model.mat');
fprintf('\nBest model: %s\n', best_model_data.model_name);
fprintf('Best model performance:\n');
fprintf('  Accuracy: %.2f%%\n', best_model_data.performance.accuracy * 100);
fprintf('  F1-Score: %.3f\n', best_model_data.performance.f1);

