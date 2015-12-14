%% convert_PCA: Perform principal component analysis on the data
function pca_reduced = convert_PCA(features, feature_parameters)
    if length(feature_parameters) == 0
        dimensions = 100;
        normalize = false;
    else
        dimensions = feature_parameters{1};
        normalize = feature_parameters{2};
    end

    data = double(features) - mean(features);

    disp 'normalizing'; fflush(stdout);
    if normalize
        data = data ./ std(data);
    end

    disp 'computing covar mat'; fflush(stdout);
    covar = cov(data);
    [U S V] = svd(covar);

    disp 'PCA analysis'; fflush(stdout);
    dimensions = min(dimensions,size(U,2));
    pca_reduced = dot(data, U(:,1:dimensions));
end
