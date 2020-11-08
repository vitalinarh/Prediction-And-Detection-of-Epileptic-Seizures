function nn = training(P_train, T_train, type, neurons, is_prediction, spec)
    
    %% Weights (Penalization)
    wp = ones(1, length(P_train));
    
    % Interical (1)
    interictal_indexes = find(T_train(1, :) == 1);
    [interictal_total, ~] = size(interictal_indexes);
    % Preictal (2)
    preictal_indexes = find(T_train(2, :) == 1);
    [preictal_total, ~] = size(preictal_indexes);
    % Ictal (3)
    ictal_indexes = find(T_train(3, :) == 1);
    [ictal_total, ~] = size(ictal_indexes);

    % Set up network error specialization for prediction
    if is_prediction == 1 % Preictal (2)
        if strcmp(spec, 'None')
            error = 1;
        % give more importance to preictal instances
        elseif strcmp(spec, 'On')
            error = (interictal_total / preictal_total);
        end
        wp(preictal_indexes) = error;
    % Set up network error specialization for detection    
    else % Ictal (3) 
        if strcmp(spec, 'None')
            error = 1;
        elseif strcmp(spec, 'On')
            error = (interictal_total / ictal_total);
        end
        wp(ictal_indexes) = error;
    end
    
    %% Configure Network
    
    if strcmp(type, "Multilayer")
        disp("Multilayer")
        net = feedforwardnet(neurons);
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.90;
        net.divideParam.valRatio = 0.10;
        net.divideParam.testRatio = 0; 
        net.trainFcn = 'trainb';
        net.trainFcn = 'traingd';
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], wp, 'UseParallel', 'no', 'UseGPU', 'no');
        
    elseif strcmp(type, "Recurrent")
        disp("Recurrent")
        net.trainFcn = 'trainb';
        net = layrecnet(1 : 2, neurons, "traingd");
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.90;
        net.divideParam.valRatio = 0.10;
        net.divideParam.testRatio = 0;
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], 'useParallel','no', 'useGPU', 'no', 'showResources', 'no');
    end
end