function outputs = nn_sigmoid_layer(inputs, weights)
  all_inputs = horzcat([1.0], inputs); % bias feature
  outputs = sigmf(all_inputs*weights);
end

function ans = sigmf(x)
  ans = 1.0 ./ (1.0 + exp(-x));
end
