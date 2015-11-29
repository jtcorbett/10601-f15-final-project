function model = build_NN(data, labels, parameters)
  if length(parameters) == 0
    classes = 10;
    hidden_layers = []; % row vector [500 500]
    learning_rate = .05;
  else
    classes = parameters{1};
    hidden_layers = parameters{2};
    learning_rate = parameters{3};
  end

  data = double(data);
  layers = horzcat([size(data, 2)], hidden_layers, [classes]);
  M = size(layers, 2)-1; % number of layers (besides input)

  weights = cell(M, 1);
  for layer_i=1:M
    % initialize a random weight in range -.5, .5 for each input and bias
    weights{layer_i} = rand(layers(layer_i) + 1, layers(layer_i+1))-.5;
  end
  old_weights = weights;
  
  iterations = 0;
  finished = 0;
  while ~finished
    iterations = iterations + 1
    
    % update the weights
    for sample_i=1:size(data, 1)
      features = data(sample_i, :);
      answer = zeros(1, classes);
      answer(1, labels(sample_i)+1) = 1;

      % calculate output for each node for each layer
      outputs = cell(M, 1);
      node_values = features;
      for layer_i=1:M
        node_values = nn_sigmoid_layer(node_values, weights{layer_i});
        outputs{layer_i} = node_values;
      end

      % calculate delta for each node for each layer
      deltas = cell(M, 1);
      deltas{M} = transpose(outputs{M} .* (1-outputs{M}) .* (answer.-outputs{M}));
      for layer_i=M-1:-1:1
        o = outputs{layer_i};
        w = weights{layer_i+1};
        d = deltas{layer_i+1};
        diff = transpose(w*d);
        deltas{layer_i} = transpose(o .* (1-o) .* diff(2:end)); % ignore bias term
      end

      % update weights
      weights{1} = weights{1} + (learning_rate * transpose(horzcat([1.0], features)) .* transpose(deltas{1}));
      for layer_i=2:M
        inputs = horzcat([1.0], outputs{layer_i-1});
        delta = deltas{layer_i};
        weights{layer_i} = weights{layer_i} + (learning_rate * transpose(inputs) .* transpose(delta));
      end
    end
    
    % evaluate performance and decide whether or not to continue
    good = zeros(size(data, 1), 1);
    for sample_i=1:size(data, 1)
      if test_NN(weights, data(sample_i, :)) == labels(sample_i)
        good(sample_i) = 1;
        % sample_i
      end
    end
    
    "\n"
    acc = sum(good)/size(data,1)
    if acc > .99
      finished = 1;
    end
    old_acc = acc;
    abs(acc - old_acc)
    sum(sum(abs(weights{1}-old_weights{1})))
    old_weights = weights;
    
    fflush(stdout);
  end
  
  model = weights;
end
