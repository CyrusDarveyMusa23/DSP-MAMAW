% Create sample features (replace with real data)
sample_features = rand(1, 16);
load('results/best_model.mat');
prediction = predict(best_model_data.model, sample_features);
fprintf('Prediction: %d (0=Clean, 1=Artifact)\n', prediction);