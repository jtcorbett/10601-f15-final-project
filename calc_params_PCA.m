function [data params] = calc_params_PCA(features, feature_parameters)
    if length(feature_parameters) == 0
        dimensions = 100;
        normalize = 0;
        whiten = 0;
    elseif length(feature_parameters) == 3
        dimensions = feature_parameters{1};
        normalize = feature_parameters{2};
        whiten = feature_parameters{3};
    else
        disp 'invalid number of args';
        return;
    end

    population_mean = mean(features);
    data = double(features) - population_mean;

    population_std = 0;
    if normalize
        disp 'normalizing';
        population_std = std(data);
        data = data ./ population_std;
    end

    disp 'computing covar mat';
    covar = cov(data)
    size(data)
    [U, S, ~] = svd(covar);

    disp 'PCA analysis';
    dimensions = min(dimensions,size(U,2));
    population_U = U(:,1:dimensions);
    data = data * population_U;

    population_white = 0;
    if whiten
        disp 'Whitening';
        population_white = sqrt(S(:,1:dimensions) + 1.0e-50);
        data = data / population_white;
    end

    params = {population_mean population_std population_U population_white};
end
