function model = build_NN(data, labels, parameters)
  if length(parameters) == 0
    classes = 10;
    hidden_layers = []; % row vector [500 500]
    learning_rate = .05;
    acc_thresh = .95;
    epoch_num = 20;
    batch_size = 100;
    eval_size = 100;
  else
    classes = parameters{1};
    hidden_layers = parameters{2};
    learning_rate = parameters{3};
    if parameters{4} >= 1
      acc_thresh = 2;
      epoch_num = parameters{4};
    else
      acc_thresh = parameters{4};
      epoch_num = 1000;
    end
    batch_size = parameters{5};
    eval_size = parameters{6};
  end
  
  
  data = double(data);
  layers = horzcat([size(data, 2)], hidden_layers, [classes]);
  L = size(layers, 2); % number of layers (besides input)
  N = size(data, 1);
  
  weights = cell(L, 1);
  best_weights = cell(L, 1);
  batch_weights = cell(L, 1);
  biases = cell(L, 1);
  best_biases = cell(L, 1);
  batch_biases = cell(L, 1);
  activation = cell(L, 1);
  output = cell(L, 1);
  deltas = cell(L, 1);
  for layer_i=1:L
    % initialize a random weight for each input and bias
    if layer_i > 1
      weights{layer_i} = normrnd(0, sqrt(layers(layer_i-1)), [layers(layer_i), layers(layer_i-1)]);
    end
    biases{layer_i} = normrnd(0, 1, [layers(layer_i), 1]);
    z{layer_i} = zeros(layers(layer_i), 1);
    output{layer_i} = zeros(layers(layer_i), 1);
    delta{layer_i} = zeros(layers(layer_i), 1);
  end
  
  %old_weights = weights;
  old_acc = 0;
  max_acc = 0;
  
  %c = onCleanup(@()clean(weights, biases));
  
  epoch_count = 0;
  for epoch_i=1:epoch_num
    epoch_i
  
    % take a random sample of data
    idx = randperm(size(data, 1));
    for batch_start=1:batch_size:size(data, 1)-batch_size
      batch_data = data(idx(batch_start:batch_start+batch_size), :);
      batch_labels = labels(idx(batch_start:batch_start+batch_size), :);

      % reset deltas for batch vars
      for layer_i=2:L
        batch_weights{layer_i} = zeros(layers(layer_i), layers(layer_i-1));
        batch_biases{layer_i} = zeros(layers(layer_i), 1);
      end  
    
      % run one batch
      for sample_i=1:batch_size
        features = batch_data(sample_i, :);
        answer = zeros(classes, 1);
        answer(batch_labels(sample_i)+1, 1) = 1;
        % answer = answer + (.5 .- answer)*.2;
        
        % calculate output for each node for each layer
        activation{1} = features';
        for layer_i=2:L
          z{layer_i} = weights{layer_i}*activation{layer_i-1} + biases{layer_i}; %'
          activation{layer_i} = sigmf(z{layer_i});
        end

        % calculate delta for each node for each layer
        deltas = cell(L, 1);
        deltas{L} = (activation{L}-answer);%(activation{L}-answer) .* sigmdiff(z{L});
        for layer_i=L-1:-1:2
          deltas{layer_i} = weights{layer_i+1}'*deltas{layer_i+1}.*(activation{layer_i}.*(1-activation{layer_i})); %(weights{layer_i+1}'*deltas{layer_i+1}) .* sigmdiff(z{layer_i}); %'
        end

        % update batch weights and biases
        for layer_i=2:L
          batch_weights{layer_i} = batch_weights{layer_i} + (deltas{layer_i}*activation{layer_i-1}'); %'
          batch_biases{layer_i} = batch_biases{layer_i} + deltas{layer_i};
        end
      end
      
      % update weights and biases
      old_weights = weights;
      for layer_i=2:L
        weights{layer_i} = weights{layer_i} - (learning_rate/batch_size)*batch_weights{layer_i};
        biases{layer_i} = biases{layer_i} - (learning_rate/batch_size)*batch_biases{layer_i};
      end
      % diff = weights{2} - old_weights{2}
    end
    
    % evaluate performance
    idx = 1:size(data, 1);%randperm(size(data, 1));
    eval_data = data(idx(1:eval_size), :);
    eval_labels = labels(idx(1:eval_size), :);
    good = zeros(1, eval_size);
    for sample_i=1:eval_size
      %out = test_NN({weights biases}, eval_data(sample_i, :))
      if test_NN({weights biases}, eval_data(sample_i, :)) == eval_labels(sample_i)
        good(sample_i) = 1;
        % "right"
        % sample_j
      end
    end
    %good
    
    acc = sum(good)/eval_size
    %if acc > max_acc
    %  best_weights = weights;
    %  best_biases = biases;
    %  max_acc = acc;
    %end
    acc_diff = acc - old_acc
    old_acc = acc;
    
    if acc > acc_thresh
      break
    end

    %sum(sum(abs(weights{2}-old_weights{2})))
    %sum(sum(abs(weights{3}-old_weights{3})))
    %sum(sum(abs(weights{4}-old_weights{4})))
    %old_weights = weights;
    
    fflush(stdout);
  end
  
  model = {weights biases};%{best_weights best_biases};
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = sigmdiff(x)
  ans = sigmf(x) .* (1-sigmf(x));
end

function clean(weights, biases)
  model={weights,biases};
  save('model.mat', 'model')
end