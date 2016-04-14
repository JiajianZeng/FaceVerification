function [ num_class ] = generate_image_list( root_dir, image_list_file )
% generate image list given a root dirctory
% parameter root_dir denotes the root directory of dataset CASIA-webface
% if parameter root_dir is ended with '/', then remove '/'
if root_dir(length(root_dir)) == '/'
    root_dir = root_dir(1:length(root_dir) - 1);
end
subfolders = dir(root_dir);
num_subfolders = length(subfolders);
file = fopen(image_list_file, 'w');
id = 1;
for i = 1:num_subfolders
    subfolder = subfolders(i).name;
    % current directory 
    if strcmp(subfolder, '.') == 1
        continue;
    end
    % parent directory
    if strcmp(subfolder, '..') == 1
        continue;
    end
    contents = dir([root_dir, '/', subfolder]);
    num_contents = length(contents);
    % empty subfolder
    if num_contents == 2
        continue;
    end
    for j = 1:num_contents
        isDir = contents(j).isdir;
        content = contents(j).name;
        if isDir == 0
            fprintf(file, '%s %d\r\n', [subfolder, '/', content], id); 
        end
        if mod(j, 10) == 0
            disp(['In subfolder ', num2str(i), '/', num2str(num_subfolders), ' Completed ', num2str(j), '/', num2str(num_contents)]);
        end
        if j == num_contents
            disp(['In subfolder ', num2str(i), '/', num2str(num_subfolders), ' Completed ', num2str(j), '/', num2str(num_contents)]);
        end
    end
    id = id + 1;
end
num_class = id - 1;
disp(['Total class:', num2str(num_class)]);
fclose(file);
end