function model = build_GNB(data, labels, parameters)
    offset = 1 - min(labels);
    labels = labels + offset;

    labels_length = length(labels);
    num_features = size(data,2);
    unique_labels = unique(labels);
    num_labels = length(unique_labels);

    preprocess_parameters = {};
    if length(parameters) > 0
      preprocess_parameters = parameters{1};
      augment = parameters{2};
    else
      augment = true;
    end
    
    if augment
      data = [data; flipLR(data)]
      labels = [labels; labels];
    end
    
    [data redo_preprocess] = preprocess(data, preprocess_parameters, {});
    
    % calculate priors
    prior = zeros(size(unique_labels));

    for label = 1:num_labels
        prior(label) = sum(labels==label)/labels_length;
    end

    for label=min(unique_labels):max(unique_labels)
      mu(label, :) = mean(data(find(labels==label), :), 1);
      variance_mat(label, :) = var(data(find(labels==label), :), 0, 1);
    end
    
    stdevs = sqrt(variance_mat);
    
    model = {offset, prior, mu, stdevs, preprocess_parameters, redo_preprocess};
end
