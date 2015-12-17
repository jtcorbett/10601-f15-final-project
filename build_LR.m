function [model] = build_LR(data, labels, parameters)

    offset = 1 - min(labels);
    labels = labels + offset;

    labels_length = length(labels);
    num_features = size(data,2);
    unique_labels = unique(labels);
    num_labels = length(unique_labels);

    preprocess_parameters = {};
    if length(parameters) > 0
      preprocess_parameters = parameters{1};
      augment = parameters{2};
    else
      preprocess_parameters = {};
      augment = true;
    end

    if augment
      data = [data; flipLR(data)];
      labels = [labels; labels];
    end

    [data redo_preprocess] = preprocess(data, preprocess_parameters, {});
    
    %
    % We hit baseline accuracy with a KNN optimization, so didn't need to finish this
    %
    

end

function [sig] = sigmoid(X):
   denom = 1.0 + exp(-1.0 * X);
   sig = 1.0 / denom;
   
function [res] = softmax(weights, biases, x):
   vec = exp((x * weights') + biases);
   res = (vec')/sum(vec, 1);
   
function [p] = predict(weights, biases, x):
    p = softmax(weights, biases, x);