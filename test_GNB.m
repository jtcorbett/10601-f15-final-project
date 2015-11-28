function category = test_GNB(model, features)
    prior = model{1};
    u = model{2};
    s = model{3};

    F = length(features);
    K = length(prior);

    category = 1;
    max_p = -inf;

    for j = 1:K
        p = log(prior(j));
        for i = 1:F
            p = p + log(normpdf(double(features(i)),u(i,j),s(i,j)));
        end

        if p > max_p
            category = j;
            max_p = p;
        end
    end
end
