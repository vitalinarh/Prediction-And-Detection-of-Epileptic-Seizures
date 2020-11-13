function [P_train, T_train, P_test, T_test] = class_balance(P, T2, Trg, train_ratio)
    indexes = [1 : length(P)]';
    temp_P = [P indexes];
    
    interictal_indexes = find(T2(:, 1));
    interictal_vector = temp_P(interictal_indexes, :);
    
    total_ictal = length(find(T2(:, 3) == 1));
    total_preictal = length(find(T2(:, 2) == 1));
    
    % Vector that indicated the begining and end of each seizure (1 and -1)
    seizures = diff(Trg);
    % Indexes with the first pre-ictal (2) of each seizure
    start_preictal = find(seizures == 1) - 900;
    % Indexes with the last pos-ictal (4) of each seizure
    end_posictal = find(seizures == -1) + 301;
    
    total_seizures = length(start_preictal);
    
    % Number of seizures present in the training data
    num_seizures_training_data = round(total_seizures * train_ratio);
    
    % Vector with preictal (2), ictal (3) and postictal (4) for the
    % training set
    seizures_idxs = [];
    for i = 1 : num_seizures_training_data
        seizures_idxs = [seizures_idxs start_preictal(i):1:end_posictal(i)];
    end
    
    % Balance dataset
    % Balance = 1: The number of inter_ictal (1) class instances is equal
    % to the sum of all other classes;
    % Balance = 10: The number of inter_ictal (1) class instances is equal
    % to ten times the sum of all other classes;
    total_interictal = length(seizures_idxs) * 1;
    
    % If the number of total interictal is larger than the existing
    % interictal instances, use only the available ones
    if(total_interictal > (length(P) - length(seizures_idxs)))
        total_interictal = length(P) - length(seizures_idxs);
    end
    
    % Choose inter_ictal indexes in a equal number to the total ones
    interictal_idxs = find(T2 == 1)';
    interictal_idxs = interictal_idxs(1 : total_interictal);
    
    % Join interinctal (1) class and the remaining classes (2,3,4 -
    % seizure)
    training_idxs = horzcat(interictal_idxs, seizures_idxs);
    training_idxs = sort(training_idxs);
    
    % Define the training set
    P_train = P(training_idxs, :)';
    T_train = T2(training_idxs, :)';   
    
    % Remaining seizures to the test set
    %P_all_idxs = 1:length(P);
    %test_idxs = setdiff(P_all_idxs, training_idxs);
    
    seizures_idxs = [];
    for i = num_seizures_training_data + 1 : total_seizures
        seizures_idxs = [seizures_idxs start_preictal(i):1:end_posictal(i)];
    end
    
    total_interictal = length(seizures_idxs) * 1;
    
    % If the number of total interictal is larger than the existing
    % interictal instances, use only the available ones
    if(total_interictal > (length(P) - length(seizures_idxs)))
        total_interictal = length(P) - length(seizures_idxs);
    end
    
    % Join interinctal (1) class and the remaining classes (2,3,4 -
    % seizure)
    test_idxs = horzcat(interictal_idxs,seizures_idxs);
    test_idxs = sort(test_idxs);

    
    % Define the test set
    P_test = P(test_idxs,:)';
    T_test = T2(test_idxs,:)';   
end