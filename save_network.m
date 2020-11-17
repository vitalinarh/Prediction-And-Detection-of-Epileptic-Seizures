function nn = save_network(nn, P_test, T_test, patient, type, is_prediction, class_balancing, train_ratio, features, neurons)

    train_ratio = int2str(train_ratio);
    neurons = int2str(neurons);
    features = int2str(features);
    
    if class_balancing == 1
        balance = 'Balance_On';
    else
        balance = 'Balance_Off';
    end 
    
    if is_prediction
        goal = 'Prediction';
    else
        goal = 'Detection';
    end
    
    filename = strcat('p', patient, '_g', goal, '_t', type, '_cl', balance, '_tr', train_ratio, '_n', neurons, '_f', features,'.mat');
    filepath = strcat('TrainedNNetworks\', filename);
    save(filepath, 'nn', 'P_test', 'T_test');
    
end