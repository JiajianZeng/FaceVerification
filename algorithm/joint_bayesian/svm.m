function [ model ] = svm( joint_bayesian_model, X, labels, ratio)
%SVM classifier for face verification
%   output argument model holds the parameters of SVM model
%   input argument joint_bayesian_model holds the parameters of joint
%   bayesian model
%   input argument X is an n by d matrix, of which each row is a feature
%   vector with dimensions d
%   input argument labels holds the corresponding label of each row in X
%   input argument ratio indicates how much the identical pair accounts for
%   to train svm

classes = unique(labels);
num_class = length(classes);

sample_in_each_class = cell(num_class, 1);
sample_num_in_each_class = zeros(num_class, 1);
for i = 1:num_class
    sample_in_each_class{i} = find(labels == i);
    sample_num_in_each_class(i) = length(sample_in_each_class{i});
end

num_sample = length(labels);
num_pair_identical = floor(num_sample / 2 * ratio);
% sample_flag indicates the state of the sample
% 0 indicates that sample is included in the pair_different
% 1 indicates that sample is included in the pair_identical
sample_flag = zeros(num_sample, 1);

% generate pair identical
pair_identical_1 = zeros(num_pair_identical * 10, 1);
pair_identical_2 = zeros(num_pair_identical * 10, 1);
pair = 1;
disp('Start generating pair identical...');
while pair < num_pair_identical
    pair_identical_in_class = randi([1, num_class], num_pair_identical * 2, 1);
    pair_identical_in_class_unique = unique(pair_identical_in_class);
    for i = 1:length(pair_identical_in_class_unique)
        class = pair_identical_in_class_unique(i);
        n = sum(pair_identical_in_class == class) * 2;
        s = sample_num_in_each_class(class);
        if n > s
            n = floor(s / 2) * 2;
        end
        idx_identical = randperm(s);
        for j = 1:n /2
            pair_identical_1(pair) = sample_in_each_class{class}(idx_identical(j*2 -1));
            pair_identical_2(pair) = sample_in_each_class{class}(idx_identical(j*2));
            pair = pair + 1;
        end    
    end
    if pair < num_pair_identical
        pair = 1;
        continue;
    end
    break;
end
disp('Completing generating pair identical...');

pair_identical_1 = pair_identical_1(1:num_pair_identical);
pair_identical_2 = pair_identical_2(1:num_pair_identical);
% update sample flag
sample_flag(pair_identical_1) = 1;
sample_flag(pair_identical_2) = 1;

% preprocess for generating pair different
num_different = sum(sample_flag == 0);
num_different = floor(num_different / 2) * 2;
different_sample = find(sample_flag == 0);

idx = randperm(num_different);
different_sample = different_sample(idx);

pair_different_1 = different_sample(1:num_different / 2);
pair_different_2 = different_sample(num_different / 2 + 1:end);

err_pair = find(labels(pair_different_1) == labels(pair_different_2));
num_err_pair = length(err_pair);

% generate pair different
% we allow to generating some identical pair when generating different pair
% which can't account for more than 1 percent
disp('Start generating pair different...');
while 2 * num_err_pair / num_different > 0.01
    idx = randperm(length(err_pair));
    pair_different_1(err_pair) = pair_different_1(err_pair(idx));
    err_pair = find(labels(pair_different_1) == labels(pair_different_2));
    num_err_pair = length(err_pair);
end
disp('Complete generating pair different...');
disp(['pair identical accounts for ', num2str(2 * num_err_pair / num_different * 100), ' percentage when generating pair different']);
disp(['num_err_pair = ', num2str(num_err_pair), ', num_different_pair = ', num2str(num_different / 2)]);

% adapt 
labels_different_pair = zeros(length(pair_different_1), 1);
for i = 1:num_different / 2
    if labels(pair_different_1(i)) == labels(pair_different_2(i))
        labels_different_pair(i) = 1;
    end
end
% compute distance 
dis_pair_identical = zeros(num_pair_identical, 1);
dis_pair_different = zeros(num_different / 2, 1);

for i = 1:num_pair_identical
    dis_pair_identical(i) = joint_bayesian_distance(joint_bayesian_model, X(pair_identical_1(i), :), X(pair_identical_2(i), :));
end

for i = 1:num_different / 2
    dis_pair_different(i) = joint_bayesian_distance(joint_bayesian_model, X(pair_different_1(i), :), X(pair_different_2(i), :));
end

data_labels = [ones(num_pair_identical, 1);labels_different_pair];
data = [dis_pair_identical;dis_pair_different];

model = svmtrain(data_labels, data, '-t 0 -h 0');
end