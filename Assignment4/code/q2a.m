f = 721.537700/10;
px = 609.559300;
py = 172.854000;
baseline = 0.5327119288*100;

K = zeros(3,3);
K(1,1) = f;
K(2,2) = f;
K(3,3) = 1;
K(1,3) = px;
K(2,3) = py;

% 
% % cite: http://matlab.wikia.com/wiki/FAQ#How_can_I_process_a_sequence_of_files.3F
% % Specify the folder where the files live.
% myFolder = '../data/test/results';
% % Check to make sure that folder actually exists.  Warn user if it doesn't.
% if ~isdir(myFolder)
%   errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
%   uiwait(warndlg(errorMessage));
%   return;
% end
% % Get a list of all files in the folder with the desired file name pattern.
% filePattern = fullfile(myFolder, '*_left_disparity.png'); % Change to whatever pattern you need.
% theFiles = dir(filePattern);
% % for k = 1 : length(theFiles)
% for k = 1:3
%   baseFileName = theFiles(k).name;
%   fullFileName = fullfile(baseFileName);
%   fprintf(1, 'Now reading %s\n', fullFileName);
%   % Now do whatever you want with this file name,
%   % such as reading it in as an image array with imread()
%   sub = regexp(fullFileName,'\d*','Match');
%   disp = getDisparity('test', string(sub));
%   hold on
%   imshow(disp)
%   depth = f*baseline./disp;
%   depth_matrix(:,:,k) = depth;
% end

data = getData([], 'test', 'list'); 
ids = data.ids(1:3);

for i = 1:length(ids)
    name = ids{i};

    disp = getDisparity('test', name)*100;
    depth = f*baseline./disp;
    figure; 
    imagesc(depth,[0,255])
    prefix = '../data/test/results/';
    ext = '.csv';
    name = strcat(name, '_', 'depth');
    savename = strcat(prefix, name, ext);
    csvwrite(savename, depth);
end
    
    