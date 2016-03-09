% script for training Joint Bayesian model and SVM model
% Joint Bayesian model is used to compute distance between two feature
% vector
% SVM model is used to classify the distance obtained by Joint Bayesian
% model into two group, which indicates whether the two feature vector
% is belonging to the same identity or not
load('lbp_WDRef.mat');
load('id_WDRef.mat');

labels = id_WDRef;
X = double(lbp_WDRef);
X = sqrt(X);
% normalize
X = bsxfun(@rdivide, X, sum(X,2));
% mean
train_mean = mean(X, 1);
% pca
[COEFF, SCORE] = princomp(X, 'econ');
train_x = SCORE(:,1:2000);

disp('training joint bayesian model...');
joint_bayesian_model = joint_bayesian(train_x, labels);
disp('complete training joint bayesian model.')

disp('training svm model...');
svm_model = svm(joint_bayesian_model, train_x, labels, 0.5);
disp('complete training svm model.');

% prepare test data 
load('lbp_lfw.mat');
load('pairlist_lfw.mat');
normX = double(lbp_lfw);
normX = sqrt(normX);
% normalize
normX = bsxfun(@rdivide, normX, sum(normX, 2));
% make sure their mean is zero vector
normX = bsxfun(@minus, normX, train_mean);
% pca
normX = normX * COEFF(:,1:2000);

test_intra = pairlist_lfw.IntraPersonPair;
test_extra = pairlist_lfw.ExtraPersonPair;

result_intra = zeros(3000, 1);
result_extra = zeros(3000, 1);
for i = 1:3000
    result_intra(i) = joint_bayesian_distance(joint_bayesian_model, normX(test_intra(i,1),:), normX(test_intra(i,2),:));
    result_extra(i) = joint_bayesian_distance(joint_bayesian_model, normX(test_extra(i,1),:), normX(test_extra(i,2),:));
end

group_sample = [ones(3000, 1);zeros(3000, 1)];
sample = [result_intra;result_extra];

[m,n] = size(sample);
predict_label = zeros(m, 1);

for i = 1:m
    value = sum(svm_model.sv_coef .* svm_model.SVs .* sample(i,1));
    value = value - svm_model.rho;
    if value > 0
        predict_label(i, 1) = 1;
    else
        predict_label(i, 1) = 0;
    end
end

disp(['accuracy = ',num2str(sum(predict_label == group_sample) / m)]);