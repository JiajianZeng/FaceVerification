function [ features ] = read_feature( file_name, dim_features)
%Read formatted features from txt file
%   Detailed explanation goes here
file = fopen(file_name);
cell = textscan(file, '%s');

num_features = size(cell{1}, 1);
features = zeros(num_features, dim_features);

disp(['Total number of features to be read, ', num2str(num_features), '.']);
for i = 1:num_features
    c = strsplit(cell{1}{i}, ',');
    for j = 1:dim_features
        features(i,j) = str2double(c{j});
    end
    if mod(i, 1000) == 0
        disp(['Read ', num2str(i), '/', num2str(num_features)]);        
    end
end
if mod(i, 1000) ~= 0
    disp(['Read ', num2str(i), '/', num2str(num_features)]);        
end
disp('Successfully read features from file.');
fclose(file);
end

