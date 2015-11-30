function model = build_NN(data, labels, parameters)
  if length(parameters) == 0
    classes = 10;
    hidden_layers = []; % row vector [500 500]
    learning_rate = .05;
    acc_thresh = .001;
    batch_size = 10;
    evaluate_thresh = 900;
  else
    classes = parameters{1};
    hidden_layers = parameters{2};
    learning_rate = parameters{3};
    acc_thresh = parameters{4};
    batch_size = parameters{5};
    evaluate_thresh = parameters{6};
  end

  data = double(data);
  layers = horzcat([size(data, 2)], hidden_layers, [classes]);
  L = size(layers, 2); % number of layers (besides input)
  N = size(data, 1);

  weights = cell(L, 1);
  batch_weights = cell(L, 1);
  biases = cell(L, 1);
  batch_biases = cell(L, 1);
  activation = cell(L, 1);
  output = cell(L, 1);
  deltas = cell(L, 1);
  for layer_i=1:L
    % initialize a random weight in range -.5, .5 for each input and bias
    if layer_i > 1
      weights{layer_i} = rand(layers(layer_i), layers(layer_i-1))-.5;
    end
    biases{layer_i} = rand(layers(layer_i), 1)-.5;
    activation{layer_i} = zeros(layers(layer_i), 1);
    output{layer_i} = zeros(layers(layer_i), 1);
    delta{layer_i} = zeros(layers(layer_i), 1);
  end

  %old_weights = weights;
  old_acc = 0;

  iterations = 0;
  strikes = 0;
  evaluate_count = 0;
  sample_i=1;
  while strikes < 3
    iterations = iterations + 1;
    batch_count = 0;

    % update the weights
    while batch_count < batch_size
      features = data(sample_i, :);
      answer = zeros(classes, 1);
      answer(labels(sample_i)+1, 1) = 1;

      % reset deltas for batch vars
      for layer_i=2:L
        batch_weights{layer_i} = zeros(layers(layer_i), layers(layer_i-1));
        batch_biases{layer_i} = zeros(layers(layer_i), 1);
      end

      % calculate output for each node for each layer
      output{1} = features';
      for layer_i=2:L
        activation{layer_i} = weights{layer_i}*output{layer_i-1} + biases{layer_i}; %'
        output{layer_i} = sigmf(activation{layer_i});
      end

      % calculate delta for each node for each layer
      deltas = cell(L, 1);
      deltas{L} = (output{L}-answer) .* sigmdiff(activation{L});
      for layer_i=L-1:-1:2
        deltas{layer_i} = (weights{layer_i+1}'*deltas{layer_i+1}) .* sigmdiff(activation{layer_i}); %'
      end

      % update batch weights and biases
      for layer_i=2:L
        batch_weights{layer_i} = batch_weights{layer_i} + (deltas{layer_i}*output{layer_i-1}'); %'
        batch_biases{layer_i} = batch_biases{layer_i} + deltas{layer_i};
      end

      if sample_i==N
        sample_i = 1;
      else
        sample_i = sample_i + 1;
      end

      batch_count = batch_count + 1;
    end

    % update weights and biases
    for layer_i=2:L
      weights{layer_i} = weights{layer_i} - (learning_rate)*batch_weights{layer_i};
      biases{layer_i} = biases{layer_i} - (learning_rate)*batch_biases{layer_i};
    end

    % evaluate performance and decide whether or not to continue
    if evaluate_count > evaluate_thresh
      iterations
      good = zeros(size(data, 1), 1);
      for sample_j=1:size(data, 1)
        if test_NN({weights biases}, data(sample_j, :)) == labels(sample_j)
          good(sample_j) = 1;
          % sample_j
        end
      end

      % when the increase in accuracy is too small, give a strike. if this happens several times, stop running
      acc = sum(good)/size(data,1)
      acc_diff = acc - old_acc
      if acc_diff < acc_thresh
        strikes = strikes + 1;
      else
        strikes = 0;
      end
      old_acc = acc;
      evaluate_count = 0;
    else
      evaluate_count = evaluate_count + 1;
    end

    %sum(sum(abs(weights{2}-old_weights{2})))
    %sum(sum(abs(weights{3}-old_weights{3})))
    %sum(sum(abs(weights{4}-old_weights{4})))
    %old_weights = weights;

    % fflush(stdout);
  end

  model = {weights biases};
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = sigmdiff(x)
  ans = sigmf(x) .* (1-sigmf(x));
end
