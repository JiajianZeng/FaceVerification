% script for training Joint Bayesian model and SVM model
% Joint Bayesian model is used to compute distance between two feature
% vector
% SVM model is used to classify the distance obtained by Joint Bayesian
% model into two group, which indicates whether the two feature vector
% is belonging to the same identity or not
load('vgg_features_fc1.mat');
load('vgg_features_fc1_p.mat');
load('vgg_labels.mat');

labels = vgg_labels;
X1 = double(vgg_features_fc1);
X2 = double(vgg_features_fc1_p);

% normalize
X1 = bsxfun(@rdivide, X1, sum(X1,2));
X2 = bsxfun(@rdivide, X2, sum(X2,2));

disp('training svm model...');
svm_model = svm_euclidean_distance(X1, X2, labels);
disp('complete training svm model.');

% weighted_sum = sum(svm_model.sv_coef .* svm_model.SVs);
% prepare test data
load('lfw_features_fc1.mat');
load('lfw_features_fc1_p.mat');
load('lfw_labels.mat');

norm_lfw_X = double(lfw_features_fc1);
norm_lfw_X_p = double(lfw_features_fc1_p);

% normalize
norm_lfw_X = bsxfun(@rdivide, norm_lfw_X, sum(norm_lfw_X, 2));
%norm_lfw_X_mean = mean(norm_lfw_X, 1);
%norm_lfw_X = bsxfun(@minus, norm_lfw_X, norm_lfw_X_mean);

% normalize
norm_lfw_X_p = bsxfun(@rdivide, norm_lfw_X_p, sum(norm_lfw_X_p, 2));
%norm_lfw_X_p_mean = mean(norm_lfw_X_p, 1);
%norm_lfw_X_p = bsxfun(@minus, norm_lfw_X_p, norm_lfw_X_p_mean);

contrast_labels = lfw_labels;
[m,n] = size(contrast_labels);

result = zeros(m, 1);

for i = 1:m
    result(i,1) = norm(norm_lfw_X(i,:) - norm_lfw_X_p(i,:));
end
[label_predicted, accuracy, dec_values] = svmpredict(contrast_labels, result, svm_model);

% the code below is for linear kernel only
predict_label = zeros(m, 1);
for i = 1:m
    value = sum(svm_model.sv_coef .* svm_model.SVs .* result(i,1));
    value = value - svm_model.rho;
    if value > 0
        predict_label(i, 1) = 1;
    else
        predict_label(i, 1) = 0;
    end
end

disp(['accuracy = ',num2str(sum(predict_label == contrast_labels) / m)]);