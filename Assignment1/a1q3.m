% Question 3

% (b) 
templates_array = readInTemplates;
% convert
cell_of_gray_templates = cellfun(@rgb2gray, templates_array, 'UniformOutput', false);
cell_of_double_templates = cellfun(@double, cell_of_gray_templates, 'UniformOutput', false);
% size(cell_of_double_images)

% (c)
display_image = imread('thermometer.png');
display_image_grayscale = rgb2gray(display_image);
display_image_double = double(display_image_grayscale);
% size(display_image_double)
[M, N] = size(display_image_double);
corrArry = cell(1,30);
for i = 1:30
     [t_H, t_W] =  size(cell_of_double_templates{1, i});
    off_set_X = round((t_W)/2)+ N -1;
    off_set_Y = round((t_H)/2)+ M -1;
    corrArry{1, i} = normxcorr2(cell_of_double_templates{1, i}, display_image_double);
end



