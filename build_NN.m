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
    if parameters{6} > 0
      eval_size = parameters{6};
    else
      eval_size = size(data, 1);
    end
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
      weights{layer_i} = normrnd(0, 1/sqrt(layers(layer_i-1)), [layers(layer_i), layers(layer_i-1)]);
    end
    biases{layer_i} = normrnd(0, 1, [layers(layer_i), 1]);
    z{layer_i} = zeros(layers(layer_i), 1);
    output{layer_i} = zeros(layers(layer_i), 1);
    delta{layer_i} = zeros(layers(layer_i), 1);
  end

  %old_weights = weights;
  old_acc = 0;
  old_e = 0;
  
  %c = onCleanup(@()clean(weights, biases));
  
  epoch_count = 0;
  for epoch_i=1:epoch_num
    "\n"
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
        answer = make_ans(classes, batch_labels(sample_i));
       
        
        % calculate output for each node for each layer
        activation{1} = features';
        for layer_i=2:L
          z{layer_i} = weights{layer_i}*activation{layer_i-1} + biases{layer_i}; %'
          activation{layer_i} = relu(z{layer_i});
        end

        % calculate delta for each node for each layer
        deltas = cell(L, 1);
        deltas{L} = (activation{L}-answer);%(activation{L}-answer) .* sigmdiff(z{L});
        for layer_i=L-1:-1:2
          deltas{layer_i} = weights{layer_i+1}'*deltas{layer_i+1}.*reludiff(z{layer_i}); %(weights{layer_i+1}'*deltas{layer_i+1}) .* sigmdiff(z{layer_i}); %'
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
    eval_labels = labels(idx(1:eval_size), 1);
    good = zeros(1, eval_size);
    e = 0;
    for sample_i=1:eval_size
      %out = test_NN({weights biases}, eval_data(sample_i, :))
      [guess output] = test_NN({weights biases}, eval_data(sample_i, :));
      ans = eval_labels(sample_i);
      e = e + hinge_error(classes, output, ans);
      if guess == ans;
        good(sample_i) = 1;
        % "right"
        % sample_j
      end
    end
    %good
    
    acc = sum(good)/eval_size
    e

    acc_diff = acc - old_acc
    old_acc = acc;
    e_diff = e - old_e
    old_e = e;
    
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

nl = sigmf

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = relu(x)
  ans = (x > 0).*x;
end

function ans = reludiff(x)
  ans = double(x > 0);
end

function err = entropy_error(c, output, ans)
  t = make_ans(c, ans);
  err = sum(t.*log(output) + (1-t).*log(1-output));
end

function err = hinge_error(c, output, ans)
  diff = (output-output(ans+1))+1;
  diff = diff .* (diff > 0);
  err = sum(diff) - 1;
end

function clean(weights, biases)
  model={weights,biases};
  save('model.mat', 'model')
end

function t = make_ans(c, n)
  t = zeros(c, 1);
  t(n+1) = 1;
  % t = t + (.5 .- t)*.2;
end
