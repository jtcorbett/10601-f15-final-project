function [results_mat] = parameter_search_KNN(td, tl, vd, vl)
    vl_cell_sizes = [8 16];
    dims = [10 25 50 100 150 200];
    K = [5 10 50];

    for norm_bool=1:-1:0
        for whiten_bool=1:-1:0
            for k = K
                for vl_cell_size_i=1:length(vl_cell_sizes)
                    for dim_i=1:length(dims);
                        if (dims == 0)
                            whiten_bool = 0;
                        end
                        vl_cell_size = vl_cell_sizes(vl_cell_size_i);
                        dim = dims(dim_i);

                        train_idx = randperm(size(td, 1), 500);
                        model = build_KNN(td(train_idx,:), tl(train_idx,:), {{0 norm_bool dim whiten_bool vl_cell_size} k});
                        idx = randperm(size(vd, 1), 100);
                        out = test_KNN(model, vd(idx, :));
                        accuracy = sum(out==vl(idx))/length(idx)
                        results_mat(dim_i, vl_cell_size_i, k, norm_bool+1, whiten_bool+1) = accuracy;
                    end
                end
                save('knn_hyperparameter_results_100.mat', 'results_mat');
                disp 'finished a cell size'; fflush(stdout);
            end
        end
    end
end
