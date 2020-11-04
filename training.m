function nn = training(P_train, T_train, type, neurons, goal)
    
    %% Configure Network
    
    if strcmp(type, "Multilayer") == 0
        net = feedforwardnet(neurons);
    end
    
    net.divideFcn = 'divideblock';
    net.divideParam.trainRatio = 0.90;
    net.divideParam.valRatio = 0.10;
    net.divideParam.testRatio = 0;  
    net.trainFcn = 'traingd';
    net.trainParam.epochs = 1000;
    nn = train(net, P_train, T_train, [],[], weights_penalization);
    
end