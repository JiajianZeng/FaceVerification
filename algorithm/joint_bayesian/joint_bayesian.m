function [ model_param ] = joint_bayesian( X, labels )
%Implementation of Joint Bayesian 
%   plz read paper "Bayesian Face Revisited: A Joint Formulation" for details
%   output argument model_param holds the parameters of the Joint Bayesian
%   model, input argument X is a m-by-d matrix, of which each row is the extracted
%   feature of an image, whose dimension is d, input argument labels holds the corresponding identity label of the
%   images

m = length(labels);
d = size(X, 2);
classes = unique(labels);
num_class = length(classes);

% instances_in_each_class is a cell array, of which each element is a
% matrix
instances_in_each_class = cell(num_class, 1);
count = 0;
number_buff = zeros(1000, 1);
% preprocess
for i = 1:num_class
    instances_in_each_class{i} = X(lables == i, :);
    num_instance = size(instances_in_each_class{i}, 1);
    if num_instance > 1
        count = count + num_instance;
    end
    if number_buff(num_instance) == 0
        number_buff(num_instance) = 1;
    end
end
disp(['Preprocess done!', 'Statistics:']);
disp(['The number of samples: m = ', num2str(m)]);
disp(['Dimensions of the features: d = ', num2str(d)]);
disp(['The number of classes: num_class = ', num2str(num_class)]);

% initialize Su and Sw
% u represents the intrinsic variable for identity
% w represents the intra-personal variable for within-person variation 
% x = u + w
tic;
u = zeros(d, num_class);
w = zeros(d, count);
j = 1;

for i = 1:num_class
    % u is the intrinsic variable for identity
    % so we compute the mean of all instances in each class to get u
    u(:, i) = mean(instances_in_each_class{i}, 1)';
    num_instance = size(instances_in_each_class{i}, 1);
    if num_instance > 1
        % w is the intra-personal variable for with-person variation
        % so we subtract the mean of all instances in each class
        % from corresponding instance in each class
        w(:, j:j + num_instance - 1) = bsxfun(@minus, instances_in_each_class{i}', u(:, i));
        j = j + num_instance;
    end
end
Su = cov(u');
Sw = cov(w');
disp('Initialize Su and Sw done!');
toc;

% optimize Su and Sw iterated
tic;
SuFG = cell(1000, 1);
SwG = cell(1000, 1);
old_Sw = Sw;

for i = 1:500
    F = inv(Sw);
    w = zeros(d, m);
    j = 1;
    for k = 1:1000
        if number_buff(k) == 1
            G = -1 .* (k .* Su + Sw) \ Su / Sw;
            SuFG{k} = Su * (F + k .* G);
            SwG{k} = Sw * G;
        end
    end
    
    for k = 1:num_class
        num_instance = size(instances_in_each_class{k}, 1);
        u(:, k) = sum(SuFG{num_instance} * instances_in_each_class{k}', 2);
        w(:, j + num_instance - 1) = bsxfun(@plus, instances_in_each_class{k}', u(:, k));
        j = j + num_instance;
    end
    Su = cov(u');
    Sw = cov(w');
    
    singular_value_ratio = norm(Sw - old_Sw) / norm(Sw);
    disp(['Optimizing Su and Sw...norm(Sw - old_Sw) / norm(Sw) = ', num2str(singular_value_ratio)]);
    if singular_value_ratio < 1e-6
        break;
    end
    old_Sw = Sw;
end
disp('Optimize Su and Sw done!');
toc;

% collect model param
F = inv(Sw);
model_param.G = -1 .* (2 * Su + Sw) \ Su / Sw;
model_param.A = inv(Su + Sw) - (F + model_param.G);
model_param.Sw = Sw;
model_param.Su = Su;
end

