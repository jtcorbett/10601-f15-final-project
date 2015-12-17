%% test_KNN: Test K-Nearest Neighbors
function category = test_KNN(model, data)
    K = model{1};
    feature_params = model{2};
    population_params = model{3};
    training_features = model{4};
    training_labels = model{5};

    features = preprocess(data, feature_params, population_params);

    N = size(features,1);
    category = zeros(N,1);

    % loop through each test samples
    for n = 1:N
        feature = features(n,:);

        % Each row represents one neighbor.
        % Col 1 is score for that neighbor.
        % Col 2 is label of that neighbor.
        KNN = inf(K,2);

        % max_k(1) is score of furthest NN.
        % max_k(2) is index of furthest NN.
        max_k = [inf 1];

        % loop through each of the training samples
        for t = 1:length(training_labels)
            score = sum(abs(feature - training_features(t,:)));
            if score < max_k(1)
                KNN(max_k(2),:) = [score training_labels(t)];
                [max_k(1) max_k(2)] = max(KNN(:,1));
            end
        end

        [M F C] = mode(KNN(:,2));
        C = C{1};
        if length(C) == 1
            category(n) = M;
        else
            min_score = inf;
            category(n) = M;
            for label = C
                for neighbor = 1:K
                    if KNN(neighbor,2) == label && KNN(neighbor,1) < min_score
                        category(n) = label;
                        min_score = KNN(neighbor,1);
                    end
                end
            end
        end
    end
end
