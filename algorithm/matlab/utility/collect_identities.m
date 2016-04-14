function [ identities ] = collect_identities( data_pair_file )
%collect identities from data pair file
file = fopen(data_pair_file);
cell = textscan(file, '%s %d');
num_identities = size(cell{2}, 1);
identities = zeros(num_identities, 1);
for i = 1 : num_identities
    identities(i, 1) = cell{2}(i);
end
end

