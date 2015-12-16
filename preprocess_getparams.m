function [processed_data preprocess_params] = preprocess_getparams(data, parameters)
    if length(parameters) == 0
        dimensions = 100;
        normalize = 0;
        whiten = 0;
    elseif length(parameters) == 3
        dimensions = parameters{1};
        normalize = parameters{2};
        whiten = parameters{3};
    else
        disp 'invalid number of args'; fflush(stdout);
        return;
    end

    population_mean = mean(data);
    data = double(data) - population_mean;

    population_std = 0;
    if normalize
        disp 'normalizing'; fflush(stdout);
        population_std = std(data);
        data = data ./ population_std;
    end

    disp 'computing covar mat'; fflush(stdout);
    covar = cov(data);
    [U, S, ~] = svd(covar);

    disp 'PCA analysis'; fflush(stdout);
    dimensions = min(dimensions,size(U,2));
    population_U = U(:,1:dimensions);
    data = data * population_U;

    population_white = 0;
    if whiten
        disp 'Whitening'; fflush(stdout);
        S = diag(S)';
        population_white = sqrt(S(1:dimensions) + 1.0e-50);
        data = data ./ population_white;
    end

    processed_data = data;
    params = {population_mean population_std population_U population_white};
end
