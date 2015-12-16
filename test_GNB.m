function category = test_GNB(model, features)
    offset = model{1};
    prior = model{2};
    mu = model{3};
    stdevs = model{4};

    features = double(features);
    
    [num_labels, num_features] = size(mu);
    
    category = 1;
    max_p = -inf;


    for label_i = 1:num_labels
        label_i
        fflush(stderr);
        p = log(prior(label_i));
        for feat_i = 1:num_features
            p = p + max(log(normpdf(features(feat_i), mu(label_i,feat_i), stdevs(label_i,feat_i))),-1e250);
        end

        if p > max_p
            category = label_i;
            max_p = p;
        end
    end

    category = category - offset;
end
