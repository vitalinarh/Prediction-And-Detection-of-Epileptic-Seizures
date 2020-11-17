function [P_train, T_train] = class_balance(P, T2)

    interictal_indexes = find(T2(:, 1) == 1);
    preictal_indexes = find(T2(:, 2) == 1);
    ictal_indexes = find(T2(:, 3) == 1);
    
    total_interictal = length(interictal_indexes);
    total_preictal = length(preictal_indexes);
    total_ictal = length(ictal_indexes);
    
    downsampled_interictal = downsample(interictal_indexes, round((total_preictal + total_ictal + total_interictal) / (total_preictal + total_ictal)));
    
    P_indexes = [downsampled_interictal; preictal_indexes; ictal_indexes];
    P_indexes = sort(P_indexes);
    
    P_train = P(P_indexes, :)';
    T_train = T2(P_indexes, :)';
    
end