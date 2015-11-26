function results = main(data, labels, k, features_id, model_id, feature_parameters, model_parameters)
	if size(data, 1) ~= size(labels, 1)
		error('Data and labels different lengths')
	end
  if ~exist('feature_parameters')
    feature_parameters = [];
  end
  if ~exist('model_parameters')
    model_parameters = [];
  end
	datasets = cross_validate(data, labels, k);
	for i=1:length(datasets)
		train_data = cell2mat(datasets(i, 1));
		test_data = cell2mat(datasets(i, 2));
		train_labels = cell2mat(datasets(i, 3));
		test_labels = cell2mat(datasets(i, 4));

		conv_fn = str2func(strcat('convert_', features_id));
		for j=1:size(test_data, 1)
			train_features(j, :) = conv_fn(train_data(j, :), feature_parameters);
			test_features(j, :) = conv_fn(test_data(j, :), feature_parameters);
		end

		build_fn = str2func(strcat('build_', model_id));
		model = build_fn(train_features, train_labels, model_parameters);

		test_fn = str2func(strcat('test_', model_id));
		for j=1:size(test_features, 1)
			results(i, j) = test_fn(model, test_features(j, :)) == test_labels(j);
		end
	end
  
  sum(results, 2) / size(results, 2)
  sum(sum(results)) / (size(results, 1) * size(results, 2))
end

% try running:
% main(data, labels, 10, 'NONE', 'SHITTY');
% 
% if your feature extractor or model have additional parameters, you can pass those in:
% main(data, labels, 10, 'NONE', 'NN', {}, {10 [500 500] .15});