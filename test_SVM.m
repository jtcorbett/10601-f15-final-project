%% test_SVM: Run an SVM classifier
function category = test_SVM(model, test_data)
    N = size(test_data,1);
    category = -inf(N, length(model));
    offset = model{1};
    singleclass_svm_models = model{2};

    for model_i = 1:length(singleclass_svm_models)
        model_i
        fflush(stdout);

        single_model = singleclass_svm_models{model_i};
        label = single_model{1};
        singleclass_svm_model = single_model{2};

        p(:,label) = test_singleclass_svm(singleclass_svm_model, test_data);
    end

    [~, category] = max(p, [], 2);
    category = category - offset;
end

function scores = test_singleclass_svm(model, test_data)
    w = model{1};
    b = model{2};
    scores = sum((w .* test_data),2) + b;
end
