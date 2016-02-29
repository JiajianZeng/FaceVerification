function [ pair1, pair2 ] = generate_data_pair( image_list_path, min_id, max_id, ratio )
% generate data pair for training or testing wiht image_id between min_id
% and max_id given the image_list_path, inclusive
image_list_file = fopen(image_list_path, 'r');
% image_list is a cell array of size n by 2, where image_list{1} holds the
% image path, and image_list{2} holds the id
image_list = textscan(image_list_file, '%s %d');
num_class = 10575;

% image_in_class holds all the indexes of the image belonging to each class 
image_in_class = cell(num_class, 1);
% image_num_in_class holds the number of the images belonging to each class
image_num_in_class = zeros(num_class, 1);

for i = 1:num_class
    image_in_class{i} = find(image_list{2} == i);
    image_num_in_class(i) = length(image_in_class{i});
end

% image id in the range [min_id, max_id], inclusive
num_sample = sum(image_list{2} <= max_id) - sum(image_list{2} < min_id);
num_pair_identical = floor(num_sample / 2 * ratio);

% image_flag denotes the state of the image
% 0 indicates that image is included in the pair_different
% 1 indicates that image is included in the pair_identical
% and -1 indicates that image is not included in any data pair
image_flag = zeros(length(image_list{2}), 1);
min_index = sum(image_list{2} < min_id);
max_index = sum(image_list{2} <= max_id);
for i = 1:length(image_flag)
    if i <= min_index
        image_flag(i) = -1;
    end
    if i > max_index
        image_flag(i) = -1;
    end
end

pair_identical_1 = zeros(num_pair_identical * 10, 1);
pair_identical_2 = zeros(num_pair_identical * 10, 1);

pair = 1;
disp('Start generating pair identical...');
while pair < num_pair_identical
    pair_identical_in_class = randi([min_id, max_id], num_pair_identical * 2, 1);
    pair_identical_in_class_unique = unique(pair_identical_in_class);
    for i = 1:length(pair_identical_in_class_unique)
        class = pair_identical_in_class_unique(i);
        n = sum(pair_identical_in_class == class) * 2;
        s = image_num_in_class(class);
        if n > s
            n = floor(s / 2) * 2;
        end
        idx_identical = randperm(s);
        for j = 1 : n / 2
            pair_identical_1(pair) = image_in_class{class}(idx_identical(j*2 - 1));
            pair_identical_2(pair) = image_in_class{class}(idx_identical(j*2));
            pair = pair + 1;
        end
    end
    if pair < num_pair_identical
        pair = 1;
        continue;
    end
    break;
end
disp('Complete generating pair identical...');

pair_identical_1 = pair_identical_1(1:num_pair_identical);
pair_identical_2 = pair_identical_2(1:num_pair_identical);

image_flag(pair_identical_1) = 1;
image_flag(pair_identical_2) = 1;

num_different = sum(image_flag == 0);
num_different = floor(num_different / 2) * 2;
different_sample = find(image_flag == 0);

idx = randperm(num_different);
different_sample = different_sample(idx);

pair_different_1 = different_sample(1:num_different / 2);
pair_different_2 = different_sample(num_different / 2 + 1:end);

err_pair = find(image_list{2}(pair_different_1) == image_list{2}(pair_different_2));
num_err_pair = length(err_pair);

% we allow to generating some identical pair when generating different pair
% which can't account for more than 1 percent
disp('Start generating pair different...');
while 2 * num_err_pair / num_different > 0.01
    idx = randperm(length(err_pair));
    pair_different_1(err_pair) = pair_different_1(err_pair(idx));
    err_pair = find(image_list{2}(pair_different_1) == image_list{2}(pair_different_2));
    num_err_pair = length(err_pair);
end
disp('Complete generating pair different...');
disp(['pair identical accounts for ', num2str(2 * num_err_pair / num_different * 100), ' percentage when generating pair different']);
disp(['num_err_pair equals ', num2str(num_err_pair), ', num_different_pair equals ', num2str(num_different / 2)]);

labels_different_pair = zeros(length(pair_different_1), 1);
for i = 1:num_different / 2
    if image_list{2}(pair_different_1(i)) == image_list{2}(pair_different_2(i))
        labels_different_pair(i) = 1;
    end
end

pair1 = [pair_identical_1;pair_different_1];
pair2 = [pair_identical_2;pair_different_2];

labels = [ones(length(pair_identical_1),1);labels_different_pair];

idx = randperm(length(pair1));
pair1 = pair1(idx);
pair2 = pair2(idx);
labels = labels(idx);

fid1 = fopen('data_pair_1.txt', 'a');
fid2 = fopen('data_pair_2.txt', 'a');
fid3 = fopen('labels.txt', 'a');

for i = 1:length(pair1)
    fprintf(fid1, '%s %d\r\n', image_list{1}{pair1(i)}, image_list{2}(pair1(i)));
    fprintf(fid2, '%s %d\r\n', image_list{1}{pair2(i)}, image_list{2}(pair2(i)));
    fprintf(fid3, '%s %d\r\n', image_list{1}{1}, labels(i));
end
fclose(image_list_file);
fclose(fid1);
fclose(fid2);
fclose(fid3);
end