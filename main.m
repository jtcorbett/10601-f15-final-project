function results = main(data, labels, k, features_id, model_id)
	if size(data, 1) ~= size(labels, 1)
		error('Data and labels different lengths')
	end
	datasets = cross_validate(data, labels, k);
	for i=1:length(datasets)
		train_data = cell2mat(datasets(i, 1));
		test_data = cell2mat(datasets(i, 2));
		train_labels = cell2mat(datasets(i, 3));
		test_labels = cell2mat(datasets(i, 4));

		conv_fn = str2func(strcat('convert_', features_id))
		for j=1:size(test_data, 1)
			train_features(j, :) = conv_fn(train_data(j, :));
			test_features(j, :) = conv_fn(test_data(j, :));
		end

		build_fn = str2func(strcat('build_', model_id))
		model = build_fn(train_features, train_labels);

		test_fn = str2func(strcat('test_', model_id))
		for j=1:size(test_features, 1)
			results(i, j) = test_fn(model, test_features(j, :)) == test_labels(j);
		end
	end
end

% try running:
% main(data, labels, 10, 'NONE', 'SHITTY')