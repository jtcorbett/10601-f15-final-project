function category = test_GNB(model, features)
    offset = model{1};
    prior = model{2};
    u = model{3};
    s = model{4};

    F = length(features);
    K = length(prior);

    category = 1;
    max_p = -inf;

    for j = 1:K
        p = log(prior(j));
        for i = 1:F
            p = p + max(log(normpdf(double(features(i)),u(i,j),s(i,j))),-1e250);
        end

        if p > max_p
            category = j;
            max_p = p;
        end
    end

    category = category - offset;
end
