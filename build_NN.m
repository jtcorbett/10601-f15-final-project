function model = build_NN(data, labels, parameters)
  if length(parameters) == 0
    classes = 10;
    hidden_layers = []; % col vector [500 500]
    learning_rate = .05;
    acc_thresh = .95;
    epoch_num = 20;
    batch_size = 100;
    eval_size = 100;
    lamb = .001; % regularization parameter
    dropout_p = .5 % percent of nodes which are dropped out
    mu = .1 % friction for momentum
    preprocess_params = {};
    augment = true;
  else
    classes = parameters{1};
    hidden_layers = parameters{2};
    learning_rate = parameters{3};
    if parameters{4} >= 1
      acc_thresh = 2;
      epoch_num = parameters{4};
    else
      acc_thresh = parameters{4};
      epoch_num = 100;
    end
    batch_size = parameters{5};
    if parameters{6} > 0
      eval_size = parameters{6};
    else
      eval_size = size(data, 1);
    end
    lamb = parameters{7}; % regularization parameter
    dropout_p = parameters{8}; % percent of nodes which are dropped out
    mu = parameters{9}; % friction for momentum
    preprocess_params = parameters{10};
    augment = parameters{11};
  end

  % model = build_NN_NEW(td, tl, {10 [500 100 50] 1 .80 100 1000 .001 .5 .1 {} true});

  % set up data
  if augment
      data = [data; flipLR(data)];
      labels = [labels; labels];
  end
  
  [data, recreate_parameters] = preprocess(data, preprocess_params, {}); % normalize data
  labels = labels + 1; % octave likes things to be 1-indexed
  
  % separate out some data for evaluation
  idx = randperm(size(data, 1));
  eval_data = data(idx(1:eval_size), :);
  eval_labels = labels(idx(1:eval_size), 1);
  data = data(idx(eval_size+1:end), :);
  labels = labels(idx(eval_size+1:end), 1);

  
  layers = horzcat([size(data, 2)], hidden_layers, [classes]);
  L = size(layers, 2); % number of layers (besides input)
  N = size(data, 1);

  weights = cell(L, 1);
  batch_weights = cell(L, 1);
  biases = cell(L, 1);
  batch_biases = cell(L, 1);
  activation = cell(L, 1);
  z = cell(L, 1);
  deltas = cell(L, 1);
  for layer_i=1:L
    % initialize a random weight for each input and 0 for each bias
    if layer_i > 1
      v{layer_i} = zeros(layers(layer_i-1), layers(layer_i));
      weights{layer_i} = normrnd(0, 1, [layers(layer_i-1), layers(layer_i)])*sqrt(2/layers(layer_i-1));
      biases{layer_i} = zeros(1, layers(layer_i));
      z{layer_i} = zeros(1, layers(layer_i));
    end
    activation{layer_i} = zeros(layers(layer_i), 1);
    delta{layer_i} = zeros(layers(layer_i), 1);
  end

  old_acc = 0;
  old_e = 99999999;
    
  epoch_count = 0;
  for epoch_i=1:epoch_num 
    "\n"
    epoch_i

    for layer_i=2:L
      dropout_mask{layer_i} = (rand(size(z{layer_i})) > dropout_p) / dropout_p;
    end  
  
    % take a random sample of data
    idx = randperm(size(data, 1));
    for batch_start=1:batch_size:size(data, 1)-batch_size
      batch_data = data(idx(batch_start:batch_start+batch_size), :);
      batch_labels = labels(idx(batch_start:batch_start+batch_size), :);
      batch_answer = make_ans(classes, batch_labels);
          
      % calculate output for each node for each layer
      activation{1} = batch_data;
      for layer_i=2:L
        z{layer_i} = activation{layer_i-1}*weights{layer_i} + biases{layer_i}; %'
        if dropout_p > 0
          z{layer_i} = z{layer_i}.*dropout_mask{layer_i};
        end
        activation{layer_i} = relu(z{layer_i});
      end
      output = sftprobs(z{L});
      as(round(batch_start/batch_size)+1) = mean(sum(output==max(output,{},2).*batch_answer, 2));
      es(round(batch_start/batch_size)+1) = mean(-log(sum(output.*batch_answer, 2)));
      
      % calculate delta for each node for each layer
      deltas = cell(L, 1);
      deltas{L} = (output-batch_answer)/batch_size;
      for layer_i=L-1:-1:2
        deltas{layer_i} = deltas{layer_i+1}*weights{layer_i+1}' .* reludiff(z{layer_i}); %'
      end

      % update batch weights and biases
      for layer_i=2:L
        batch_weights{layer_i} = lamb*weights{layer_i} + (activation{layer_i-1}'*deltas{layer_i}); %'
        batch_biases{layer_i} = sum(deltas{layer_i});
      end
      
      % update weights and biases
      for layer_i=2:L
        % update weights with momentum term
        v{layer_i} = mu*v{layer_i} - (learning_rate/batch_size)*batch_weights{layer_i};
        weights{layer_i} = weights{layer_i} + v{layer_i};
        biases{layer_i} = biases{layer_i} - (learning_rate/batch_size)*batch_biases{layer_i};
      end
    end
    
    % evaluate performance
    good = zeros(1, eval_size);
    for sample_i=1:eval_size
      output = feedforward(eval_data(sample_i, :), weights, biases);
      [M guess] = max(output);
      % if sample_i == 1 output end
      answ = eval_labels(sample_i);
      if guess == answ
        good(sample_i) = 1;
        % "right"
        % sample_j
      end
    end
    %good
    
    e = mean(es) + reg_loss(weights, lamb);
    acc_internal = mean(as)
    acc = sum(good)/eval_size
    e

    old_acc = acc;
    e_diff = e - old_e
    old_e = e;
    
    if acc_internal > acc_thresh
      break
    end

    % annealing
    if mod(epoch_i, 10) == 1
      learning_rate = learning_rate*.5;
      %mu = (mu + 1-mu)/3;
      if learning_rate <= 0.00001
        break
      end
    end
    
    %sum(sum(abs(weights{2}-old_weights{2})))
    %sum(sum(abs(weights{3}-old_weights{3})))
    %sum(sum(abs(weights{4}-old_weights{4})))
    %old_weights = weights;

    fflush(stdout);
  end
  
  model = {weights biases preprocess_params recreate_parameters};
  parameters
  
end

nl = sigmf

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = relu(x)
  ans = (x >= 0).*x + (x < 0).*x*.01;
end

function ans = reludiff(x)
  ans = double(x >= 0) + double(x < 0)*.01;
end

function err = entropy_loss(c, output, answ)
  t = make_ans(c, answ);
  err = sum(t.*log(output) + (1-t).*log(1-output));
end

function err = hinge_loss(c, output, answ)
  diff = (output-output(answ+1))+1;
  diff = diff .* (diff > 0);
  err = sum(diff) - 1;
end

function err = softmax_loss(c, output, answ)
  err = -log(output(answ));
end

function err = reg_loss(w, lamb)
  err = 0;
  for i=1:size(w)
    err = err + .5*lamb*sum(sum(w{i}.*w{i}));
  end
end

function err = reg_gradient(w, lamb)
  err = lamb*w;
end

function ans = softmax_gradient(c, output, answ)
  probs = sftprobs(output);
  ans = probs - (make_ans(c, answ))';
end

function probs = sftprobs(vec)
  expd = exp(vec);
  probs = expd./sum(expd, 2);
end

function clean(weights, biases)
  model={weights,biases};
  save('model.mat', 'model')
end

function t = make_ans(c, labels)
  t = zeros(size(labels,1), c);
  for i=1:size(labels,1)
    t(i, labels(i)) = 1;
  end
end
