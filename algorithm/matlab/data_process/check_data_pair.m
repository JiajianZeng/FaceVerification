function [ flag ] = check_data_pair( pair_data_1_path, pair_data_2_path, labels_path )
% check whether the pair data in pair_data_1 and pair_data_2 is consistent
% with label in labels
flag = true;

fid1 = fopen(pair_data_1_path, 'r');
fid2 = fopen(pair_data_2_path, 'r');
fid3 = fopen(labels_path, 'r');

pair_data_1 = textscan(fid1, '%s %d');
pair_data_2 = textscan(fid2, '%s %d');
labels = textscan(fid3, '%s %d');

num_pair_data_1 = length(pair_data_1{2});
num_pair_data_2 = length(pair_data_2{2});
num_labels = length(labels{1});
num_pair_identical = 0;
num_pair_different = 0;

if num_pair_data_1 == num_pair_data_2 && num_pair_data_2 == num_labels
    for i = 1:num_pair_data_1
        if pair_data_1{2}(i) == pair_data_2{2}(i) && labels{2}(i) == 1  
            num_pair_identical = num_pair_identical + 1;
            continue;
        elseif pair_data_1{2}(i) ~= pair_data_2{2}(i) && labels{2}(i) == 0
            num_pair_different = num_pair_different + 1;
            continue;
        else
            flag = false;
            break;
        end
    end
else
    flag = false;
end

fclose(fid1);
fclose(fid2);
fclose(fid3);

if flag == 1
    result = 'correct';
else
    result = 'incorrect';
end

disp(['The pair data is ' result]);
disp('Statistics:');
disp(['Total pair:', num2str(num_pair_data_1)]);
disp(['Pair identical:', num2str(num_pair_identical)]);
disp(['Pair different:', num2str(num_pair_different)]);
end

