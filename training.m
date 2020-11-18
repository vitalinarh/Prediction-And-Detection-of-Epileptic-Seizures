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
            error = (interictal_total / preictal_total) * 2;
        else
            error = (interictal_total / preictal_total);
        end 
        wp(preictal_indexes) = error;
    % Set up network error specialization for detection, Ictal (3) 
    elseif is_prediction == false
        if strcmp(spec, 'None')
            error = 1;
        else
            if strcmp(class_balancing, 'Off')
                error = (interictal_total / ictal_total) * 2;
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
        net.divideParam.trainRatio = 0.70;
        net.divideParam.valRatio = 0.15; 
        net.divideParam.testRatio = 0.15; 
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], wp, 'UseParallel', 'no', 'UseGPU', 'no');
        
    elseif strcmp(type, "Recurrent")
        disp("Recurrent")
        net = layrecnet(1 : 2, neurons, "trainscg");
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.90;
        net.divideParam.valRatio = 0.10;
        net.divideParam.testRatio = 0;
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], 'useParallel', 'no', 'useGPU', 'no');
        
    elseif strcmp(type, "LSTM") 
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
        numHiddenUnits = 10;
        numClasses = 4;
        miniBatchSize = 27;
        maxEpochs = 50;
        
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
        miniBatchSize = 27;
        maxEpochs = 50;
        
        [P_train, T_train] = convert_vector_4d(P_train, T_train);
        
        T_train = categorical(T_train);
        
        layers = [
            imageInputLayer([29 29 1])
            
            convolution2dLayer(3, 8, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2, 'Stride', 2)
            
            convolution2dLayer(3, 16, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2, 'Stride', 2)
            
            convolution2dLayer(3,32,'Padding','same')
            batchNormalizationLayer
            reluLayer
                
            fullyConnectedLayer(3)
            softmaxLayer
            classificationLayer
        ];

        options = trainingOptions('adam',...
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs', maxEpochs, ...
            'Verbose', false, ...
            'Plots', 'training-progress');

        nn = trainNetwork(P_train, T_train, layers, options);

    end
end