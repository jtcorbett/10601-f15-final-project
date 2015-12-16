function [category output] = test_NN(model, features)
  weights = model{1};
  biases = model{2};
  preprocessing = model{3};
  means = preprocessing{1};

  features = (double(features)-means)./127.0;

  L = size(weights, 1);

  z = cell(L, 1);
  activation = cell(L, 1);

  activation{1} = features;
  for layer_i=2:L
    z{layer_i} = activation{layer_i-1}*weights{layer_i} + biases{layer_i}; %'
    activation{layer_i} = relu(z{layer_i});
  end
  output = sftprobs(z{L});

  [M I] = max(output);

  category = uint8(I) - 1;
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end

function ans = relu(x)
  ans = (x > 0).*x;
end

function probs = sftprobs(vec)
  expd = exp(vec);
  probs = expd/sum(expd, 2);
end
