function [prediction_results, detection_results]  = testing(nn, P_test, T_test, classification, type)
        
    %% Treatment for Deep NN
    if strcmp(type, "LSTM") || strcmp(type, "CNN")
        
        P_test = num2cell(P_test, 1);
        miniBatchSize = 29;
        T_Pred = classify(nn, P_test, ...
            'MiniBatchSize', miniBatchSize, ...
            'SequenceLength', 'longest');
        
        res = zeros(length(T_Pred), 4);
        for i = 1 : length(T_Pred)
            res(i, T_Pred(i)) = 1;
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
    
    [~, col] = size(res);
    
    for i = 1 : col
        max_ind = 1;
        for j = 1 : 3
            if(res(j, i) >= res(max_ind, i))
                max_ind = j;
            end
        end
        for j = 1 : 4
            res(j, i) = 0;
        end
        res(max_ind, i) = 1; 
    end
    
    %% GROUPED CLASSIFICATION TREATMENT
    if strcmp(classification, 'Grouped')
        
        threshold = 10;
        consecutives = 5;
        
        for i = 1 : (length(res) - threshold)
            counter = 0;
            for j = i : i + threshold
                % check if is equal to the next point
                if res(:, i) == res(:, j)
                    counter = counter + 1;
                end
            end
            % didn't reach the , make it interictal
            if(counter < consecutives)
                res(1, i) = 1;
                res(2, i) = 0;
                res(3, i) = 0;
            end
        end
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
    
    [~, col] = size(res);
    
    % Perform the calculations
    for i = 1 : col
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
    % Sensitivity, TP / (TP + FN)
    prediction_results.sensitivity = prediction_results.true_positives / (prediction_results.true_positives + prediction_results.false_negatives);
    % Specificity, TN / (FP + TN)
    prediction_results.specificity = prediction_results.true_negatives / (prediction_results.true_negatives + prediction_results.false_positives);
    % Accuracy, (TP + TN) / (TP + TN + FP + FN)
    prediction_results.accuracy = (prediction_results.true_positives + prediction_results.true_negatives) / (prediction_results.true_positives + prediction_results.true_negatives + prediction_results.false_positives + prediction_results.false_negatives);
   
    % Sensitivity, TP / (TP + FN)
    detection_results.sensitivity = detection_results.true_positives / (detection_results.true_positives + detection_results.false_negatives);
    % Specificity, TN / (FP + TN)
    detection_results.specificity = detection_results.true_negatives / (detection_results.true_negatives + detection_results.false_positives);
    % Accuracy, (TP + TN) / (TP + TN + FP + FN)
    detection_results.accuracy = (detection_results.true_positives + detection_results.true_negatives) / (detection_results.true_positives + detection_results.true_negatives + detection_results.false_positives + detection_results.false_negatives);
end
