function [] = data_statistics( image_list_path )
% do some statistics of the data
% first statistic we want to know is the distribution of the data
image_list_file = fopen(image_list_path, 'r');
image_statistics_file = fopen('image_statistics.txt', 'w');
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

fprintf(image_statistics_file, '%s\r\n', '######################distribution######################');
fprintf(image_statistics_file, '%s   %s   %s\r\n', 'id range', 'percentage', 'amount');
total = length(image_list{2});
percentage = 0.01;
num_image = 0;
for i = 1:num_class
    num_image = num_image + image_num_in_class(i);
    if num_image / total >= percentage
        record = ['[1,', num2str(i), ']   ', num2str(num_image / total * 100), '%   ', num2str(num_image)];
        fprintf(image_statistics_file, '%s\r\n', record);
        percentage = percentage + 0.01;
    else
        continue;
    end
end
fprintf(image_statistics_file, '%s\r\n', '######################distribution######################');

fclose(image_list_file);
fclose(image_statistics_file);
end

