function [train_data train_labels valid_data valid_labels] = createBigData()
  load subset_CIFAR10/small_data_batch_1.mat
  bdata = data;
  blabels = labels;
  load subset_CIFAR10/small_data_batch_2.mat
  bdata = vertcat(bdata, data);
  blabels = vertcat(blabels, labels);
  load subset_CIFAR10/small_data_batch_3.mat
  bdata = vertcat(bdata, data);
  blabels = vertcat(blabels, labels);
  load subset_CIFAR10/small_data_batch_4.mat
  train_data = vertcat(bdata, data);
  train_labels = vertcat(blabels, labels);
  
  load subset_CIFAR10/small_data_batch_5.mat
  valid_data = vertcat(bdata, data);
  valid_labels = vertcat(labels, labels);
end