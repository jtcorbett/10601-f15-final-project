function [results_mat] = parameter_search_GNB(td, tl, vd, vl)
  vl_cell_sizes = [8 4 16];
  dims = [0 10 25 50 100 150 200 300 500 750 1000];
  
  for norm_bool=1:-1:0
  for whiten_bool=1:-1:0
  for vl_cell_size_i=1:3
  for dim_i=1:11
    if (dims == 0) whiten_bool = 0; end
    vl_cell_size = vl_cell_sizes(vl_cell_size_i);
    dim = dims(dim_i);
    
    model = build_GNB(td, tl, {{0 norm_bool dim whiten_bool vl_cell_size} false});
    out = test_GNB(model, vd);
    accuracy = sum(out==vl);
    results(dims, vl_cell_size, norm_bool, whiten) = accuracy;
  end
  
  end
end