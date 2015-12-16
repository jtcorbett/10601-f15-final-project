function category = test_all_GNB(model, data)
    offset = model{1};
    prior = model{2};
    mu = model{3};
    stdevs = model{4};

    data = double(data);
    
    [num_labels, num_features] = size(mu);
    
    category = 1;
    max_p = -inf;


    for label_i = 1:num_labels
        label_i
        fflush(stdout);
        p(:, label_i) = log(prior(label_i))*ones(size(data, 1), 1);
        for feat_i = 1:num_features
            p(:, label_i) = p(:, label_i) + max(log(normpdf(data(:, feat_i), mu(label_i,feat_i), stdevs(label_i,feat_i))),-1e250);
        end
    end
    
    [M category] = max(p, [], 2);

    category = category - offset;
   
end
