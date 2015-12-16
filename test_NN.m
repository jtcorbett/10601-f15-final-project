function [category] = test_NN(model, data)
  weights = model{1};
  biases = model{2};
  preprocess_params = model{3};
  recreate_params = model{4};

  data = preprocess(data, preprocess_params, recreate_params);

  for input_i=1:size(data, 1)
    output = feedforward(data(input_i, :), weights, biases)
    [M I] = max(output);
    category(input_i, 1) = I;
  end

  category = uint8(category) - 1;
end