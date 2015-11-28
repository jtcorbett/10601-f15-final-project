function gnb_model = build_GNB(data, labels)
    N = length(labels);
    F = size(data,2);
    Y = unique(labels);
    K = length(Y);

    % calculate priors
    prior = zeros(size(Y));

    for label = 1:K
        prior(label) = sum(labels==label)/N;
    end

    u = zeros(F,K);
    s = zeros(F,K);

    for i = 1:F
        for j = 1:K

            % calculate mean
            u(i,j) = 1/sum(labels==j) * sum(X(find(labels==j),i));

            % calculate variance
            for n = 1:N
                if labels(n) == j
                    s(i,j) = (X(n,i) - u(i,k))^2;
                end
            end

            % normalize variance and convert to std. dev.
            s(i,j) = sqrt(s(i,j)/sum(labels==j));
        end
    end

    gnb_model = {prior, u, s};
end
