%% test_KNN: Test K-Nearest Neighbors
function category = test_KNN(model, features)
    K = model{1};
    N = size(features,1);
    training_data = model{2};
    training_labels = model{3};
    category = zeros(N,1);

    % loop through each test samples
    for n = 1:N
        epoch_i = n
        fflush(stdout);

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
            score = sum(abs(feature - training_data(t,:)));
            if score < max_k(1)
                KNN(max_k(2),:) = [score training_labels(t)];
                [max_k(1) max_k(2)] = max(KNN(:,1));
            end
        end

        category(n) = mode(KNN(:,2));
    end
end
