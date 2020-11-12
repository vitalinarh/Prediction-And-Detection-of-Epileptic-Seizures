function [prediction_results, detection_results]  = testing(nn, P_test, T_test, classification, type)
        
    %% Treatment for Deep NN
    if strcmp(type, "LSTM")
        
        miniBatchSize = 29;
        TPred = classify(nn, P_test, ...
            'MiniBatchSize', miniBatchSize, ...
            'SequenceLength', 'longest');
        
        res = zeros(length(TPred), 4);
        for i = 1 : length(res)
            res(i, TPred(i)) = 1; 
        end
        
        res = res';
    else
        %% Simulate the test normally
        res = sim(nn, P_test);
    end
    
    %% Post Processing
    % The prediction goes for the max value, so we change it to 1 and the rest is 0's
    % Go through all the results, one by one and in each iteration see
    % which class has the highest value, then assign 1 to it, the rest is
    % 0's
    
    [~, rows] = size(res);
    
    for i = 1 : rows
        max_ind = 1;
        for j = 1 : 4
            if(res(j, i) >= res(max_ind, i))
                max_ind = j;
            end
        end
        for j = 1 : 4
            res(j, i) = 0;
        end
        res(max_ind, i) = 1; 
    end
    
    %% Calculate the results
    % Setup the prediction results and the detection results
    prediction_results.true_positives = 0;
    prediction_results.true_negatives = 0;
    prediction_results.false_positives = 0;
    prediction_results.false_negatives = 0;
    prediction_results.sensitivity = 0;
    prediction_results.specificity = 0;
    prediction_results.accuracy = 0;
    
    detection_results.true_positives = 0;
    detection_results.true_negatives = 0;
    detection_results.false_positives = 0;
    detection_results.false_negatives = 0;
    detection_results.sensitivity = 0;
    detection_results.specificity = 0;
    detection_results.accuracy = 0;
    
    [~, rows] = size(res);
    
    % Perform the calculations
    for i = 1 : rows
        % Prediction - 2nd Column
        if(res(2, i) == T_test(2, i))
            % If the result of the trained network got the preictal point right, it is a
            % true positive
            if(res(2, i) == 1)
                prediction_results.true_positives = prediction_results.true_positives + 1;
            % If it isn't a preictal point and the network got it right, it
            % is a true negative
            else
                prediction_results.true_negatives = prediction_results.true_negatives + 1;
            end
        else
            % If it didn't get it right and predicted a preictal instance in the wrong point
            % it is a false positive
            if(res(2, i) == 1)
                prediction_results.false_positives = prediction_results.false_positives + 1;
            % Predicted a preictal point in a non preictal instance, false
            % negative
            else
                prediction_results.false_negatives = prediction_results.false_negatives + 1;
            end
        end
        % Detection, follows same logic as the prediction
        if(res(3, i) == T_test(3, i))
            if(res(3, i) == 1)
                detection_results.true_positives = detection_results.true_positives + 1;
            else
                detection_results.true_negatives = detection_results.true_negatives + 1;
            end
        else
            if(res(3, i) == 1)
                detection_results.false_positives = detection_results.false_positives + 1;
            else
                detection_results.false_negatives = detection_results.false_negatives + 1;
            end
        end
    end
    
    %% Perform the calculations
    % Sensitivity
    prediction_results.sensitivity = prediction_results.true_positives / (prediction_results.true_positives + prediction_results.false_negatives);
    % Specificity
    prediction_results.specificity = prediction_results.true_negatives / (prediction_results.true_negatives + prediction_results.false_positives);
    % Accuracy
    prediction_results.accuracy = (prediction_results.true_positives + prediction_results.true_negatives) / (prediction_results.true_positives + prediction_results.true_negatives + prediction_results.false_positives + prediction_results.false_negatives);
   
    % Sensitivity
    detection_results.sensitivity = detection_results.true_positives / (detection_results.true_positives + detection_results.false_negatives);
    % Specificity
    detection_results.specificity = detection_results.true_negatives / (detection_results.true_negatives + detection_results.false_positives);
    % Accuracy
    detection_results.accuracy = (detection_results.true_positives + detection_results.true_negatives) / (detection_results.true_positives + detection_results.true_negatives + detection_results.false_positives + detection_results.false_negatives);
    
    
end
