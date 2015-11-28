function gnb_model = build_GNB(data, labels)
    offset = 1 - min(labels);
    labels = labels + offset;

    N = length(labels);
    F = size(data,2);
    Y = unique(labels);
    K = max(Y);

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
            u(i,j) = 1/sum(labels==j) * sum(data(find(labels==j),i));

            % calculate variance
            for n = 1:N
                if labels(n) == j
                    s(i,j) = s(i,j) + (data(n,i) - u(i,j))^2;
                end
            end

            % normalize variance and convert to std. dev.
            s(i,j) = sqrt(s(i,j)/sum(labels==j));
        end
    end

    gnb_model = {offset, prior, u, s};
end
