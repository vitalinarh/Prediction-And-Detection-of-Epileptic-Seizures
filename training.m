function nn = training(P_train, T_train, type, neurons, is_prediction, spec, class_balancing)
    
    %% Weights (Penalization)
    wp = ones(1, length(P_train));
    
    % Interical (1)
    interictal_indexes = find(T_train(1, :) == 1);
    [~, interictal_total] = size(interictal_indexes);
    % Preictal (2)
    preictal_indexes = find(T_train(2, :) == 1);
    [~, preictal_total] = size(preictal_indexes);
    % Ictal (3)
    ictal_indexes = find(T_train(3, :) == 1);
    [~, ictal_total] = size(ictal_indexes);

    % Set up network error specialization for prediction, Preictal (2)
    if is_prediction 
        if strcmp(class_balancing, 'Off')
            error = (interictal_total / preictal_total) * 1.5;
        else
            error = (interictal_total / preictal_total);
        end 
        wp(preictal_indexes) = error;
    % Set up network error specialization for detection, Ictal (3) 
    else 
        if strcmp(spec, 'None')
            error = 1;
        else
            if strcmp(class_balancing, 'Off')
                error = (interictal_total / ictal_total) * 1.5;
            else
                error = (interictal_total / ictal_total);
            end
        end
        wp(ictal_indexes) = error;
    end
    
    %% Configure Network
    
    if strcmp(type, "Multilayer")
        disp("Multilayer")
        net = feedforwardnet(neurons, 'traingd');
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.90;
        net.divideParam.valRatio = 0.10; 
        net.divideParam.testRatio = 0.0; 
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], wp, 'UseParallel', 'no', 'UseGPU', 'no');
        
    elseif strcmp(type, "Recurrent")
        disp("Recurrent")
        net.trainFcn = 'traingd';
        net = layrecnet(1 : 2, neurons, "trainscg");
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.90;
        net.divideParam.valRatio = 0.10;
        net.divideParam.testRatio = 0;
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], 'useParallel', 'no', 'useGPU', 'no', 'showResources', 'no');
        
    elseif strcmp(type, "LSTM")
        % https://www.mathworks.com/help/deeplearning/ug/classify-sequence-data-using-lstm-networks.html   
        % Set up the cell arrays for training
        
        % create a 1-by-length(P_train) cell array, where each cell contains a 29-by-1 column of P_train.
        P_train = num2cell(P_train, 1);
        
        class1Ind = T_train(1, :) == 1;
        class2Ind = T_train(2, :) == 1;
        class3Ind = T_train(3, :) == 1;
        
        T_train = zeros(1, length(T_train)); 
        T_train(class1Ind) = 1;
        T_train(class2Ind) = 2;
        T_train(class3Ind) = 3;
        
        % T_train is a categorical vector of labels.
        T_train = categorical(T_train)';
       
        % Set up the Network
        numFeatures = 29;
        numHiddenUnits = 100;
        numClasses = 3;
        miniBatchSize = 27;
        maxEpochs = 20;
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits, 'OutputMode', 'last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        
        options = trainingOptions('adam', ...
                                'ExecutionEnvironment', 'cpu', ...
                                'GradientThreshold', 1, ...
                                'MaxEpochs', maxEpochs, ...
                                'MiniBatchSize', miniBatchSize, ...
                                'SequenceLength', 'longest', ...
                                'Shuffle','never', ...
                                'Verbose', 0, ...
                                'Plots', 'training-progress');
                            
        nn = trainNetwork(P_train, T_train, layers, options); 
    else
        disp("CNN")

    end
end