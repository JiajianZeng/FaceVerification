% rootDir = 'E:/FaceDatasets/CASIA_WebFace_all/Normalized_Faces/webface/100/';
% generate_image_list(rootDir);

%around 800W pairs
for i = 1: 4
    [pair1, pair2] = generate_data_pair('image_list.txt', 1, 10575, 0.3);
end
