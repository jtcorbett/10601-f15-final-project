function category = test_NN(model, features)
  M = size(model, 1);
  weights = model;
  node_values = double(features);
  
  for layer_i=1:M
    node_values = nn_sigmoid_layer(node_values, weights{layer_i});
  end
  
  [M I] = max(node_values);
  
  category = uint8(I) - 1;
end