function out = classify1(model, data)
  for i=1:size(data)
    out(i, 1) = test_GNB(model, data(i, :));
    i
    fflush(stdout);
  end
end