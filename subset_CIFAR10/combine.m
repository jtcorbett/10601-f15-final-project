%% combine: combines batches in the args
function [data, labels] = combine(batches)
    data = [];
    labels = [];
    for i = batches
        filename = strcat('small_data_batch_', num2str(i), '.mat');
        S = load(filename);
        data = [data ; S.data];
        labels = [labels ; S.labels];
    end
end
