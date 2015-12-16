%% build_KNN: K-Nearest Neighbors
function model = build_KNN(data, labels, parameters)
    if length(parameters) == 0
        K = 5;
    else
        K = parameters{1};
    end

    model = {K data labels};
end
