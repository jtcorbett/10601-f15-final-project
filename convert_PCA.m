%% convert_PCA: Perform principal component analysis on the data
function [pca_reduced data U] = convert_PCA(features, feature_parameters)
    if length(feature_parameters) == 0
        dimensions = 100;
        normalize = 0;
        whiten = 0;
        data = double(features) - mean(features);
    elseif length(feature_parameters) == 5
        dimensions = feature_parameters{1};
        normalize = feature_parameters{2};
        whiten = feature_parameters{3};
        population_mean = feature_parameters{4};
        population_std = feature_parameters{5};
        data = double(features) - population_mean;
    else
        disp 'invalid number of args';
        return;
    end

    if normalize
        disp 'normalizing';
        data = data ./ std(data);
    end

    disp 'computing covar mat';
    covar = cov(data)
    size(data)
    [U, S, ~] = svd(covar);

    disp 'PCA analysis';
    dimensions = min(dimensions,size(U,2));
    pca_reduced = data * U(:,1:dimensions);

    if whiten
        disp 'Whitening';
        pca_reduced = pca_reduced / sqrt(S(:,1:dimensions) + 1.0e-50);
    end
end
