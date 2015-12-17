%% build_KNN: K-Nearest Neighbors
function model = build_KNN(data, labels, parameters)
    if length(parameters) == 0
        K = 5;
        feature_params = {0 1 100 1 1};
    elseif length(parameters) == 2
        K = parameters{1};
        feature_params = parameters{2};
    else
        disp('Incorrect number of parameters');
        return;
    end

    [features, population_params] = preprocess(data, feature_params, {});

    model = {K feature_params population_params features labels};
end
