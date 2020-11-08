function [prediction_results, detection_results]  = testing(nn, P_test, T_test, classification)

    res = sim(nn, P_test);
    
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
    
end