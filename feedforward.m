function [output] = feedforward(inputs, weights, biases)
  L = size(weights, 1);

  z = cell(L, 1);
  activation = cell(L, 1);

  activation{1} = inputs;
  for layer_i=2:L
    z{layer_i} = activation{layer_i-1}*weights{layer_i} + biases{layer_i}; %'
    activation{layer_i} = relu(z{layer_i});
  end
  output = sftprobs(z{L});

end

function probs = sftprobs(vec)
  expd = exp(vec);
  probs = expd/sum(expd, 2);
end

function ans = relu(x)
  ans = (x > 0).*x;
end