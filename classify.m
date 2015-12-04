function out = classify(model, data)
  for i=1:size(data)
    out(i, 1) = test_NN(model, data(i, :));
  end
end