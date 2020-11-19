function [P_train2, T_train2] = convert_vector_4d(P_train, T_train)

    % Indexes of each class
    class1Ind = find(T_train(1, :) == 1);
    class2Ind = find(T_train(2, :) == 1);
    class3Ind = find(T_train(3, :) == 1);
    
    num_Img = 1;
    % CONVERT P_train INTO A 4D MATRIX
    % INTER-ICTAL
    for i = 1 : 29 : length(class1Ind) - 29  
        aux = [];
        for j = i : i + 28
            aux = [aux P_train(:, class1Ind(j))];
        end
        if num_Img == 1
            P_train2 = aux;
            T_train2 = 1;
            num_Img = num_Img + 1;
        else
            [~, col] = size(aux);
            if col == 29
                P_train2(:, :, 1, num_Img) = aux;
                T_train2 = [T_train2; 1];
                num_Img = num_Img + 1;
            end     
        end 
    end
        
   % PRE-ICTAL
   for i = 1 : 29 : length(class2Ind) - 29
       aux = [];
       for j = i : i + 28
           aux = [aux P_train(:, class2Ind(j))];
       end
       if num_Img == 1
           P_train2 = aux;
           T_train2 =  2;
           num_Img = num_Img + 1;
       else
           [~, col] = size(aux);
           if col == 29
               P_train2(:, :, 1, num_Img) = aux;
               T_train2 = [T_train2;  2];
               num_Img = num_Img + 1;
           end
       end
   end
        
    % ICTAL
    for i = 1 : 29 : length(class3Ind) - 29
        aux = [];
        for j = i : i + 28
            aux = [aux P_train(:, class3Ind(j))];
        end
        if num_Img == 1
            P_train2 = aux;
            T_train2 =  3;
            num_Img = num_Img + 1;
        else
            [~, col] = size(aux);
            if col == 29
                P_train2(:, :, 1, num_Img) = aux;
                T_train2 = [T_train2; 3];
                num_Img = num_Img + 1;
            end     
        end
    end
end