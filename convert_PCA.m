%% convert_PCA: Perform principal component analysis on the data
function [pca_reduced] = convert_PCA(features, feature_parameters, population_parameters)
    if length(feature_parameters) == 0
        dimensions = 100;
        normalize = 0;
        whiten = 0;
    elseif length(feature_parameters) == 3
        dimensions = feature_parameters{1};
        normalize = feature_parameters{2};
        whiten = feature_parameters{3};
    else
        disp 'invalid number of args for feature_parameters'; fflush(stdout);
        return;
    end
    if length(population_parameters) == 4
        population_mean = population_parameters{1};
        population_std = population_parameters{2};
        population_U = population_parameters{3};
        population_whiten = population_parameters{4};
        data = double(features) - population_mean;
    elseif length(population_parameters) > 0
        disp 'invalid number of args for population_parameters'; fflush(stdout);
        return;
    else
        data = double(features) - mean(features);
    end

    if normalize
        disp 'normalizing'; fflush(stdout);
        if length(population_parameters) == 4
            data = data ./ population_std;
        else
            data = data ./ std(data);
        end
    end

    disp 'computing covar mat'; fflush(stdout);
    if length(population_parameters) == 4
        U = population_U;
        S = population_whiten;
    else
        covar = cov(data);
        [U, S, ~] = svd(covar);
        S = sqrt(S(:,1:dimensions) + 1.0e-50);
    end

    disp 'PCA analysis'; fflush(stdout);
    dimensions = min(dimensions,size(U,2));
    pca_reduced = data * U(:,1:dimensions);

    if whiten
        disp 'Whitening'; fflush(stdout);
        pca_reduced = pca_reduced / S;
    end
end
