function [results_mat] = parameter_search_GNB(td, tl, vd, vl)
    vl_cell_sizes = [4 8 16];
    dims = [0 10 25 50 100 150 200 300 500 750 1000];

    for norm_bool=1:-1:0
        for whiten_bool=1:-1:0
            for vl_cell_size_i=1:length(vl_cell_sizes)
                for dim_i=1:length(dims)
                    if (dims == 0)
                        whiten_bool = 0;
                    end

                    vl_cell_size = vl_cell_sizes(vl_cell_size_i);
                    dim = dims(dim_i);

                    model = build_GNB(td, tl, {{0 norm_bool dim whiten_bool vl_cell_size} false});
                    idx = randperm(size(vd, 1),1000);
                    out = test_GNB(model, vd(idx, :));
                    accuracy = sum(out==vl(idx))/length(idx);
                    results_mat(dim_i, vl_cell_size_i, norm_bool+1, whiten_bool+1) = accuracy;
                end
                save('gnb_hyperparameter_results.mat', 'results_mat');
                disp 'finished a cell size'; fflush(stdout);
            end
        end
    end
end
