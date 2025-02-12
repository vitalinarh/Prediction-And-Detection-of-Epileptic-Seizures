function [P_train, P_test, T_train, T_test] = build(dataset, features, train_ratio, classes, class_balancing)

    %% LOAD DATASET
    %  PATIENT 1
    if strcmp(dataset, "1")
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
    
    if classes == 3
        T(ictal_indices(1) - 900 : ictal_indices(1) - 1) = 2; % pre-ictal, 900 points before the first 1 for each 
                                                              % seizure, corresponding to 900 seconds = 15 minutes.
        % Run through all indices
        for i = 2 : length(ictal_indices)
           before_index = ictal_indices(i - 1);
           current_index = ictal_indices(i);
           for j = current_index - 900 : current_index - 1
               % if not ictal instance, make it preictal
               if T(j) == 0
                   T(j) = 2;
               end 
           end  
           
           for j = before_index + 1 : before_index + 301
               % if not ictal instance, make it post ictal
               if T(j) == 0
                   T(j) = 4;
               end 
           end
        end

        T(current_index + 1 : current_index + 301) = 4;       % Take care of the last seizure, posictal not changed in the loop

        T(T == 1) = 3; % Ictal, change 1 to 3
        T(T == 0) = 1; % Interictal, rest of 0 values is 1

        % Perform transformation from one column to four columns for each class
        T2 = zeros(length(T), 3);
        
    elseif classes == 2
        T(T == 1) = 1; % Ictal
        T(T == 0) = 2; % Non-ictal
        
        T2 = zeros(length(T), 2);
    end

    for i = 1 : length(T)
       T2(i, T(i)) = 1; 
    end
    
    %% REDUCE THE NUMBER OF FEATURES using an autoencoder
    X = data.FeatVectSel;
    if features < 29
        hiddenSize = features;
        autoenc1 = trainAutoencoder(X.', hiddenSize, ...
        'MaxEpochs', 200, ...
        'EncoderTransferFunction','satlin', ...
        'DecoderTransferFunction','purelin', ...
        'L2WeightRegularization',0.001, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.05, ...
        'ScaleData', false); 
        feat1 = encode(autoenc1, X.');
        P = feat1.';
    else
        P = data.FeatVectSel;
    end

    %% Define the Training set and the Testing set
    if class_balancing == 1
        %% TO DO
        Q = length(P);

        [trainInd, testInd] = divideblock(Q, train_ratio,  1 - train_ratio);
        
        P_train = P(trainInd, :);
        T_train = T2(trainInd, :);
        
        [P_train, T_train] = class_balance(P_train, T_train);
        
        P_test = P(testInd, :)';
        T_test = T2(testInd, :)';
        
    % No Class Balancing
    else
        if train_ratio == 0
           
            % Training set
            P_train = P';
            T_train = T2';

            % Test set
            P_test = P';
            T_test = T2';
            
        else
            Q = length(P);

            [trainInd, testInd] = divideblock(Q, train_ratio,  1 - train_ratio);
            
            % Training set
            P_train = P(trainInd, :)';
            T_train = T2(trainInd, :)';

            % Test set
            P_test = P(testInd, :)';
            T_test = T2(testInd, :)';
            
        end
    end
end
