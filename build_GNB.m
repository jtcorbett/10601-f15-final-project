function model = build_GNB(data, labels, parameters)
    offset = 1 - min(labels);
    labels = labels + offset;

    labels_length = length(labels);
    num_features = size(data,2);
    unique_labels = unique(labels);
    num_labels = length(unique_labels);

    % calculate priors
    prior = zeros(size(unique_labels));

    for label = 1:num_labels
        prior(label) = sum(labels==label)/labels_length;
    end

    mu = zeros(num_features,num_labels);
    variance_mat = zeros(num_features,num_labels);

    for i = 1:num_features
        for j = 1:num_labels

            % calculate mean
            mu(i,j) = 1/sum(labels==j) * sum(data(find(labels==j),i));

            % calculate variance
            for n = 1:labels_length
                if labels(n) == j
                    variance_mat(i,j) = variance_mat(i,j) + (data(n,i) - mu(i,j))^2;
                end
            end

            % normalize variance and convert to std. dev.
            variance_mat(i,j) = sqrt(variance_mat(i,j)/sum(labels==j));
        end
    end

    model = {offset, prior, mu, variance_mat};
end
