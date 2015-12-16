function category = test_GNB(model, features)
    offset = model{1};
    prior = model{2};
    mu = model{3};
    variance_mat = model{4};

    num_features = length(features);
    num_labels = length(prior);

    category = 1;
    max_p = -inf;

    for j = 1:num_labels
        p = log(prior(j));
        for i = 1:num_features
            p = p + max(log(normpdf(double(features(i)),mu(i,j),variance_mat(i,j))),-1e250);
        end

        if p > max_p
            category = j;
            max_p = p;
        end
    end

    category = category - offset;
end
