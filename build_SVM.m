%% build_SVM: Builds an SVM classifier
function svn_model = build_SVM(train_data, train_labels, parameters)
    if length(parameters) == 0
        C = 10;
    elseif length(parameters) == 1
        C = parameters{1};
    else
        disp('Invalid number of model parameters');
        return;
    end

    train_data = double(train_data);
    train_labels = double(train_labels);

    offset = 1 - min(train_labels);
    train_labels = train_labels + offset;
    features = unique(train_labels)';
    singleclass_svn_models = cell(size(features));

    for f = 1:length(features)
        epoch_label = f
        fflush(stdout);
        single_labels = 2*(train_labels==features(f))-1;
        singleclass_model = build_singleclass_SVM(train_data, single_labels, C);
        singleclass_svn_models{f} = {features(f) singleclass_model};
    end

    svn_model = {offset singleclass_svn_models};
end

% Build a model for a single class SVM (train_labels must be 1 or -1)
function singleclass_svn_model = build_singleclass_SVM(train_data, train_labels, C)
    N = length(train_labels);
    H = (train_labels * train_labels') .* (train_data * train_data');
    f = -ones(N,1);
    tic();
    % alpha = qp([],H,f,train_labels',0,zeros(N,1),C*ones(N,1));
    alpha = qp([],H,f,train_labels',0,zeros(N,1),[]);
    toc();
    w = sum((alpha .* train_labels) .* train_data);
    S_i = find(alpha>0)';
    b = sum(train_labels(S_i) - sum((alpha(S_i)' .* train_labels(S_i)') .* H(S_i,S_i),2))/sum(alpha>0);
    singleclass_svn_model = {w};

    % uncomment to include the constant offest in the model
    % singleclass_svn_model = {w b};
end

% kernal for X, only guarenteed to work for positive values
function phi = kernel(X)
    % pairwise_terms is the pairwise producs of all elements in X
    pairwise_terms = bsxfun(@times,X,X');
    pairwise_terms = sqrt(2)*(pairwise_terms(triu(true(size(pairwise_terms)),1))');
    phi = [1 (sqrt(2)*X) (X .^ 2) pairwise_terms];
end

function dot_product = kernel_mult(X,Y)
    dot_product = (X'*Y + 1)^2;
end
