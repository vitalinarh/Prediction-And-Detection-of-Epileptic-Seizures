function [P_train, T_train] = class_balance(P, T2, Trg, train_ratio)
    indexes = [1 : length(P)]';
    temp_P = [P indexes];
    
    interictal_indexes = find(T2(:, 1));
    interictal_vector = temp_P(interictal_indexes, :);
    
    total_ictal = length(find(T2(:, 3) == 1));
    ictal_indexes = find(T2(:, 3));
    ictal_vector = temp_P(ictal_indexes, :);
    
    total_preictal = length(find(T2(:, 2) == 1));
    preictal_indexes = find(T2(:, 2));
    preictal_vector = temp_P(preictal_indexes, :);
    
    downsampled_interictal = downsample(interictal_vector, round(length(interictal_vector) / (total_ictal + total_preictal)));   
    P_new = [];
    T_new = [];
    
    [~, cols] = size(downsampled_interictal);
    j = 1;
    
    for i = 1 : length(downsampled_interictal)
        if downsampled_interictal(i, cols) > ictal_vector(j, cols) && j < total_ictal
            P_new = [P_new; ictal_vector(j, 1 : cols)];
            T_new = [T_new; T2(ictal_vector(j, cols), 1 : 3)];
            j = j + 1;
        else
            P_new = [P_new; downsampled_interictal(i, 1 : cols)]; 
            T_new = [T_new; T2(downsampled_interictal(i, cols), 1 : 3)];
        end
    end
    
    j = 1;
    P_new2 = [];
    T_new2 = [];
    
    for i = 1 : length(P_new)
        if P_new(i, cols) > preictal_vector(j, cols) && j < total_preictal
            P_new2 = [P_new2; preictal_vector(j, 1 : cols)];
            T_new2 = [T_new2; T2(preictal_vector(j, cols), 1 : 3)];
            j = j + 1;
        else
            P_new2 = [P_new2; P_new(i, 1 : cols)];
            T_new2 = [T_new2; T2(P_new(j, cols), 1 : 3)];
        end
    end
    
    P_train = P_new2(:, 1 : cols - 1);
    T_train = T_new2(:,:);
end