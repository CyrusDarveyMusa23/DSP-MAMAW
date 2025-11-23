function train_xgboost(Xtrain, Ytrain, Xtest, Ytest)
    % 1. Ensure data is double (prevents categorical/string errors)
    Xtrain = double(Xtrain);
    Ytrain = double(Ytrain);
    Xtest  = double(Xtest);
    Ytest  = double(Ytest);

    % 2. Convert to Python Numpy Arrays
    py_Xtrain = py.numpy.array(Xtrain);
    py_Ytrain = py.numpy.array(Ytrain);
    py_Xtest  = py.numpy.array(Xtest);
    py_Ytest  = py.numpy.array(Ytest);
    
    % Import XGBoost
    xgb = py.importlib.import_module('xgboost');
    
    % 3. FIX: Pass Label as the 2nd argument (Positional)
    % DO NOT pass the string 'label'
    dtrain = xgb.DMatrix(py_Xtrain, py_Ytrain);
    dtest  = xgb.DMatrix(py_Xtest,  py_Ytest);
    
    % Define Parameters
    params = py.dict(pyargs( ...
        'objective', 'binary:logistic', ...
        'eval_metric', 'logloss', ...
        'max_depth', int32(6), ...      % Explicit int32 is safer for Python
        'eta', 0.1, ...
        'subsample', 0.8, ...
        'colsample_bytree', 0.8, ...
        'nthread', int32(4) ...
    ));
    
    % Train
    num_round = int32(100); % Ensure integer
    model = xgb.train(params, dtrain, num_round);
    
    % Predict
    pred = model.predict(dtest);
    
    % Convert Python prediction back to MATLAB double
    pred_double = double(pred);
    
    % Threshold
    pred_label = double(pred_double > 0.5);
    
    % Accuracy
    accuracy = sum(pred_label' == Ytest) / length(Ytest); % Transpose pred_label just in case
    fprintf('XGBoost Accuracy: %.2f%%\n', accuracy * 100);
    
    % Plot
    figure;
    confusionchart(Ytest, pred_label);
    title('XGBoost Confusion Matrix');
end