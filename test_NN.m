function category = test_NN(model, features)
  weights = model{1};
  biases = model{2};
  L = size(weights, 1);

  activation = cell(L, 1);
  output = cell(L, 1);  
  
  output{1} = double(features)';
  for layer_i=2:L
    activation{layer_i} = weights{layer_i}*output{layer_i-1} + biases{layer_i}; %'
    output{layer_i} = sigmf(activation{layer_i});
  end

  %outlayer = output{L}'
  [M I] = max(output{L});

  category = uint8(I) - 1;
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end
