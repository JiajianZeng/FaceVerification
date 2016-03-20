function [ ] = mat2yaml( variable, opencv_mat_name, data_type, filename, mode)
%convert .mat file to .yaml file 

[rows,cols] = size(variable);
% linear indexing
variable = variable';

file = fopen(filename, mode);
if mode == 'w'
    fprintf(file, '%%YAML:1.0\n');
end

% write variable header
disp(['##########', opencv_mat_name])
disp(['start writing header for ', opencv_mat_name, '...']);

fprintf(file, '   %s: !!opencv-matrix\n', opencv_mat_name);
disp(['opencv-matrix:', opencv_mat_name]);

fprintf(file, '       rows: %d\n', rows);
disp(['rows:', num2str(rows)]);

fprintf(file, '       cols: %d\n', cols);
disp(['cols:', num2str(cols)]);

fprintf(file, '       dt: %s\n', data_type);
disp(['data type:', data_type]);

disp(['complete writing header for ', opencv_mat_name]);

% write variable data
disp(['start writing data for ', opencv_mat_name, '...']);
fprintf(file, '       data: [ ');
for i = 1:rows * cols
    fprintf(file, '%.6f', variable(i));
    if mod(i, rows) == 0
        disp(['finished:', num2str(i / rows), '/', num2str(cols), '(cols)']);
    end
    if(i == rows * cols)
        break;
    end
    fprintf(file, ',');
    if mod(i+1, 7) == 0
        fprintf(file, '\n            ');
    end 
end
disp(['complete writing data for ', opencv_mat_name]);
disp(['##########', opencv_mat_name]);
fprintf(file, ']\n');
fclose(file);

end



