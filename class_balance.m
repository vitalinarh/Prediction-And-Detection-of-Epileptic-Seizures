function [P_train, T_train, P_test, T_test] = class_balance(P, T2, Trg, train_ratio)
    indexes = [1 : length(P)]';
    temp_P = [P indexes];
    
    interictal_indexes = find(T2(:, 1));
    interictal_vector = temp_P(interictal_indexes, :);
    
    total_ictal = length(find(T2(:, 3) == 1));
    total_preictal = length(find(T2(:, 2) == 1));
    
    
end