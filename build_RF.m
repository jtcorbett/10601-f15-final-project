function model = build_RF(train, labels, n, k)
    % build multiple trees and store in cell array
end

function tree_model = build_tree(train, labels)
    Y = unique(labels);

    if size(Y,1) == 0
    % handle case of no labels

    elseif size(Y,1) == 1
    % humongous labels
        tree_model = [Y(1)];
    end

    % select feature that maximizes gain
    F = size(train,2);
    selected = 1;
    max_gain = 0;

    for f = 1:F
        g = gain(labels, train(:,f));
        if g > max_gain
            selected = f;
            max_gain = g;
        end
    end

    pos_indicies = find(train(:,selected)==1);
    neg_indicies = find(train(:,selected)==0);

    tree_model = [
        selected ;
        build_tree(train(pos_indicies,:), labels(pos_indicies)) ;
        build_tree(train(neg_indicies,:), labels(neg_indicies))
    ];

    labels = labels(labels ~= selected)
end

function I = gain(X,Y)
    I = entropy(X) - entropyY(X,Y);
end

function H = entropyY(X,Y)
    n = unique(Y);
    s = size(n,1);
    H = zeros(s,1);

    for v = 1:s
        H(v) = (size(find(Y==n(v)),1) / size(Y,1)) * entropy(X(Y==n(v)));
    end

    H = sum(H);
end

function H = entropy(X)
    n = unique(X);
    s = size(n,1);
    H = zeros(s,1);

    for x = 1:s
        H(x) = size(find(X==n(x)),1) / size(X,1);
        H(x) = -H(x)*log2(H(x));
    end

    H = sum(H);
end
