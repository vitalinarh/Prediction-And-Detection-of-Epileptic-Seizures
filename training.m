function nn = training(P_train, T_train, type, neurons, is_prediction)
    
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
        nn = train(net, P_train, T_train);
    elseif strcmp(type, "Recurrent")
        disp("Recurrent")
        net.trainFcn = 'trainb';
        net = layrecnet(1 : 2, neurons, "traingd");
        net.divideFcn = 'divideind';
        [net.divideParam.trainInd, ~, ~] = divideind(length(P_train), 1:length(P_train), [],[]);
        net.trainParam.epochs = 1000;
        nn = train(net, P_train, T_train, [], [], 'useParallel','no', 'useGPU', 'no', 'showResources', 'no');
    end
end