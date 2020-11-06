function [P_train, P_test, T_train, T_test] = build(dataset, features, train_ratio)

    %% LOAD DATASET
    %  PATIENT 1
    if strcmp(dataset, "Patient 1") == 0
        data = load("Datasets/112502.mat");
    % PATIENT 2
    else
        data = load("Datasets/44202.mat");
    end
    
    %% CONVERT TARGET OF DATASET
    %  1's to 3 - Ictal
    %  0's to 1's, 2's and 4's - Interictal, Preictal, Posical,
    %  respectively
    T = data.Trg;
    
    ictal_indices = find(T == 1); % Get indices of 1's, Ictal indices
    
    T(ictal_indices(1) - 900 : ictal_indices(1) - 1) = 2; % 900 points before the first 1 for each 
                                                          % seizure, corresponding to 900 seconds=15 minutes.
    % Run through all indices
    for i = 2 : length(ictal_indices)
        before_idx = ictal_indices(i - 1);
        current_index = ictal_indices(i);
        T(current_index - 900 : current_index - 1) = 2; % pre-ictal
        T(before_idx + 1 : before_idx + 301) = 4; % pos-ictal
    end
    
    T(current_index + 1 : current_index + 301) = 4;       % Take care of the last seizure, posictal not changed in the loop
    
    T(T == 1) = 3; % Ictal, change 1 to 3
    T(T == 0) = 1; % Interictal, rest of 0 values is 1
    
    % Perform transformation from one column to four columns for each class
    T2 = zeros(length(T), 4);
    for i = 1 : length(T)
       T2(i, T(i)) = 1; 
    end
    
    %% REDUCE THE NUMBER OF FEATURES using an autoencoder
    X = data.FeatVectSel;
    if features < 29
        hiddenSize = features;
        autoenc1 = trainAutoencoder(X(1 : 10000, :).', hiddenSize, ...
        'MaxEpochs', 400, ...
        'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.15, ...
        'ScaleData', false); 
        feat1 = encode(autoenc1, X.');
        P = feat1.';
    else
        P = data.FeatVectSel;
    end 
    
    % P = reduce_dataset(data.FeatVectSel, data.Trg, features);
    
    %% Define the Training set and the Testing set
    
    % Last index of the last posictal instance
    posictal = find(T2(:,4) == 1);
    last_index = posictal(end);
        
    % Keep only data before that instance
    P = P(1 : last_index, :);
    Q = length(P);
    % valRatio = 0; % No validation data
    test_ratio =  1 - train_ratio;
    [trainInd,testInd] = divideblock(Q, train_ratio, test_ratio);
        
    % Define the training set
    P_train = P(trainInd, :)';
    T_train = T2(trainInd, :)';
        
    % Define the test set
    P_test = P(testInd, :)';
    T_test = T2(testInd, :)';
end