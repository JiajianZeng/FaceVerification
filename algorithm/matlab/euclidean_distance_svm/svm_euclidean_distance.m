function [ model ] = svm_euclidean_distance( feature_mat1, feature_mat2, contrastive_label )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
num_features = size(feature_mat1, 1);
num_labels = size(contrastive_label, 1);
if num_features < num_labels
    m = num_features;
else
    m = num_labels;
end

distances = zeros(m, 1);

labels = contrastive_label(1:m, 1);
for i = 1:m
    distances(i, 1) = norm(feature_mat1(i,:) - feature_mat2(i,:));
end

% for more details about the options used for svmtrain, plz refer to homepage of LibSVM  
model = svmtrain(labels, distances, ['-h 0 -c 16 -g ',num2str(2^1)]);

end

