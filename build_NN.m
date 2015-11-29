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
  L = size(layers, 2); % number of layers (besides input)
  N = size(data, 1);
  
  weights = cell(L, 1);
  biases = cell(L, 1);
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
  finished = 0;
  while ~finished
    iterations = iterations + 1
    
    % update the weights
    for sample_i=1:N
      features = data(sample_i, :);
      answer = zeros(classes, 1);
      answer(labels(sample_i)+1, 1) = 1;

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

<<<<<<< HEAD
      % update weights and biases
      for layer_i=2:L
        weights{layer_i} = weights{layer_i} - (learning_rate)*(deltas{layer_i}*output{layer_i-1}'); %'
        biases{layer_i} = biases{layer_i} - (learning_rate)*deltas{layer_i};
      end
    end
    
    % evaluate performance and decide whether or not to continue
    good = zeros(size(data, 1), 1);
    for sample_j=1:size(data, 1)
      if test_NN({weights biases}, data(sample_j, :)) == labels(sample_j)
        good(sample_j) = 1;
        % sample_j
      end
=======
    % calculate delta for each node for each layer
    deltas = cell(M, 1);
    deltas{M} = transpose(outputs{M} .* (1 - outputs{M}) .* (answer - outputs{M}));
    for layer_i=M-1:-1:1
      o = outputs{layer_i};
      w = weights{layer_i+1};
      d = deltas{layer_i+1};
      diff = transpose(w*d);
      deltas{layer_i} = transpose(o .* (1-o) .* diff(2:end)); % ignore bias term
    end

    % update weights
    weights{1} = weights{1} + (learning_rate * transpose(horzcat([1.0], features)) * transpose(deltas{1}));
    for layer_i=2:M
      inputs = horzcat([1.0], outputs{layer_i-1});
      delta = deltas{layer_i};
      weights{layer_i} = weights{layer_i} + (learning_rate * transpose(inputs) * transpose(delta));
>>>>>>> 6946eed6b9c20358d6f3132ee1cd6d4d039de9c9
    end
    
    acc = sum(good)/size(data,1)
    if acc > .45
      finished = 1;
    end
    acc_diff = acc - old_acc
    old_acc = acc;
    
    %sum(sum(abs(weights{2}-old_weights{2})))
    %sum(sum(abs(weights{3}-old_weights{3})))
    %sum(sum(abs(weights{4}-old_weights{4})))
    %old_weights = weights;
    
    fflush(stdout);
  end
  
  model = {weights biases};
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = sigmdiff(x)
  ans = sigmf(x) .* (1-sigmf(x));
end
