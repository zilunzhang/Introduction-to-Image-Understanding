addpath(genpath(pwd))

detector_name_list = {'detector-car', 'detector-person', 'detector-bicycle'};

color_list = {'r', 'b', 'g'};

data = getData([], 'test', 'list'); 

ids = data.ids(1:3);

for j = 1:length(ids)
    for i = 1:3
        detector_name = detector_name_list{1,i};

        col = color_list{1,i};

        name = ids{j,1};

        data = getData([], [], detector_name);

        model = data.model;

        test_train = 'test'; 

        left_right = 'left';


        imdata = getData(name, test_train, left_right);
        im = imdata.im;
        f = 1.5;
        imr = imresize(im,f); % if we resize, it works better for small objects

% detect objects
        fprintf('running the detector, may take a few seconds...\n');
        tic;

        if i == 1 && j == 1
            [ds, bs] = imgdetect(imr, model, model.thresh+0.55); % you may need to reduce the threshold if you want more detections
        end
        if i == 2 && j == 1
            [ds, bs] = imgdetect(imr, model, model.thresh+0.95); % you may need to reduce the threshold if you want more detections
        end
        if i == 3 && j == 1
            [ds, bs] = imgdetect(imr, model, model.thresh+2.0); % you may need to reduce the threshold if you want more detections
        end
        
        if i == 1 && j == 2
            [ds, bs] = imgdetect(imr, model, model.thresh+0.65); % you may need to reduce the threshold if you want more detections
        end
        if i == 2 && j == 2
            [ds, bs] = imgdetect(imr, model, model.thresh+1.0); % you may need to reduce the threshold if you want more detections
        end
        if i == 3 && j == 2
            [ds, bs] = imgdetect(imr, model, model.thresh+0.4); % you may need to reduce the threshold if you want more detections
        end
        
        if i == 1 && j == 3
            [ds, bs] = imgdetect(imr, model, model.thresh+0.7); % you may need to reduce the threshold if you want more detections
        end
        if i == 2 && j == 3
            [ds, bs] = imgdetect(imr, model, model.thresh+2.0); % you may need to reduce the threshold if you want more detections
        end
        if i == 3 && j == 3
            [ds, bs] = imgdetect(imr, model, model.thresh+2.0); % you may need to reduce the threshold if you want more detections
        end
        e = toc;
        fprintf('finished! (took: %0.4f seconds)\n', e);
        nms_thresh = 0.5;
        top = nms(ds, nms_thresh);
        if model.type == model_types.Grammar
        bs = [ds(:,1:4) bs];
        end
        if ~isempty(ds)
            % resize back
            ds(:, 1:end-2) = ds(:, 1:end-2)/f;
            bs(:, 1:end-2) = bs(:, 1:end-2)/f;
        end;
        if ~isempty(bs)
            figure; showboxesMy(im, reduceboxes(model, bs(top,:)), col);
            fprintf('detections:\n');
            ds = ds(top, :);

            prefix = '../data/test/results/';
            ext = '.csv';
            name = strcat(name, '_', detector_name);
            savename = strcat(prefix, name, ext);
            csvwrite(savename, ds);
        else
            figure;
            image(im)
        end
    end
end