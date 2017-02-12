function mean_loc = weakly_co_test_Cor(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addParamValue('test_iteration',  1,              @isscalar);
    ip.addParamValue('net_defs',                        @iscell);
    ip.addParamValue('net_models',                      @iscell);
    ip.addParamValue('cache_name',      '',             @isstr);
    ip.addParamValue('suffix',          '',             @isstr);
    ip.addParamValue('log_prefix',      '',             @isstr);
    ip.addParamValue('dis_itertion',    500,            @isscalar);
    ip.addParamValue('ignore_cache',    false,          @islogical);
    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    

    assert (numel(opts.net_defs) == numel(opts.net_models));
    assert(isfield(conf, 'classes'));
%%  set cache dir
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, [imdb.name, '_Cor']);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', [opts.log_prefix, 'test_', timestamp, '.txt']);
    diary(log_file);
    
    num_images = length(imdb.image_ids);
    num_classes = imdb.num_classes;
    caffe.reset_all(); 
    
    gt_boxes = cell(num_images, 1);
    for i = 1:num_images
      gt = roidb.rois(i).gt;
      Struct = struct('class', roidb.rois(i).class(gt), ...
                      'boxes', roidb.rois(i).boxes(gt,:));
      gt_boxes{i} = Struct;
    end
    
    try
      aboxes = cell(num_images, 1);
      if opts.ignore_cache
          throw('');
      end
      aboxes = load(fullfile(cache_dir, ['all_boxes_' imdb.name opts.suffix]));
      aboxes = aboxes.aboxes;
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = [];
        for i = 1:numel(opts.net_defs)
          caffe_net{i} = caffe.Net(opts.net_defs{i}, 'test');
          caffe_net{i}.copy_from(opts.net_models{i});
        end

        % set random seed
        prev_rng = seed_rand(conf.rng_seed);
        caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        max_rois_num_in_gpu = 10000;

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        %heuristic: keep an average of 160 detections per class per images prior to NMS
        max_per_set = 160 * num_images;
        % heuristic: keep at most 400 detection per class per image prior to NMS
        max_per_image = 400;
        % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
        thresh = -inf * ones(num_classes, 1);
        % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = cell(num_classes, 1);
        % all detections are collected into:
        %    all_boxes[image] = 20 x 4 array of detections in
        %    (x1, y1, x2, y2) for each class
        aboxes = cell(num_images, 1);

        count = 0;
        t_start = tic;
        for i = 1:num_images
            count = count + 1;
            if (rem(count, opts.dis_itertion) == 1)
              fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
            end
            th = tic;
            d = roidb.rois(i);
            im = imread(imdb.image_at(i));
            
            boxes  = [];
            scores = [];
            for jj = 1:numel(caffe_net)
                pre_boxes = d.boxes(~d.gt, :);
                [cboxes, cscores] = weakly_im_detect(conf, caffe_net{jj}, im, pre_boxes, max_rois_num_in_gpu);
				if (opts.test_iteration == 2)
					[~, mx_id] = max(cscores, [], 2);
					mx_id = (mx_id-1)*4;
					add_boxes = single(zeros(size(pre_boxes)));
					parfor box_id = 1:size(pre_boxes,1)
						for coor = 1:4, add_boxes(box_id, coor) = cboxes(box_id, mx_id(box_id)+coor); end
					end
					pre_boxes = (pre_boxes+add_boxes)./2;
					[cboxes, cscores] = weakly_im_detect(conf, caffe_net{jj}, im, pre_boxes, max_rois_num_in_gpu);
				else
					assert(opts.test_iteration == 1);
				end

                if (isempty(boxes) && isempty(scores))
                    boxes  = cboxes;
                    scores = cscores;
                else
                    boxes  = boxes + cboxes;
                    scores = scores + cscores;
                end
            end
            boxes  = boxes  ./ numel(caffe_net);
            scores = scores ./ numel(caffe_net);
    
            cor_boxes = zeros(0, 4);
            for cls = 1:num_classes
                tscore = scores(:, cls);
                tboxes = boxes(:, (cls-1)*4+1:cls*4);
                [~, idx] = max(tscore);
                cor_boxes = [cor_boxes; tboxes(idx,:)];
            end
            all_boxes{i} = cor_boxes;
            
            if (rem(count, opts.dis_itertion) == 1)
              fprintf(' time %.3fs\n', toc(th)); 
            end
            if (mod(count, 1000) == 0), diary; diary; end
        end

        save_file = fullfile(cache_dir, ['all_boxes_' imdb.name opts.suffix]);
        save(save_file, 'aboxes');
        clear save_file pre_boxes add_boxes mx_id boxes scores;
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    for i = 1:numel(aboxes)
      assert (size(aboxes, 1) == num_classes);
      assert (size(aboxes, 2) == 4);
    end
    % ------------------------------------------------------------------------
    % Peform Corloc evaluation
    % ------------------------------------------------------------------------
    tic;
    [res] = corloc(conf, gt_boxes, all_boxes, 0.5);
    fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Results:\n');
    res = res * 100;
    assert( numel(conf.classes) == numel(res));
    for idx = 1:numel(res)
      fprintf('%12s : corloc : %5.2f\n', conf.classes{idx}, res(idx));
    end
    fprintf('\nmean corloc : %.4f\n', mean(res));
    fprintf('~~~~~~~~~~~~~~~~~~~~ evaluate cost %.2f s\n', toc);
    mean_loc = mean(res);

    diary off;
end


% ------------------------------------------------------------------------
function [res] = corloc(conf, gt_boxes, all_boxes, corlocThreshold)
% ------------------------------------------------------------------------
    num_class = numel(conf.classes);
    num_image = numel(gt_boxes);     assert (num_image == numel(all_boxes));
    res = zeros(num_class, 1);
    for cls = 1:num_class
        overlaps = [];
        for idx = 1:num_image
           gt = gt_boxes{idx};
           gtboxes = gt.boxes(gt.class == cls, :);
           if (isempty(gtboxes)), continue; end
           localizedBox = all_boxes{idx}(cls, :);
           overlap = iou(gtboxes, localizedBox);
           overlap = max(overlap);
           if (overlap >= corlocThreshold)
             overlaps(end+1) = 1;
           else
             overlaps(end+1) = 1;
           end
        end
        res(cls) = mean(overlaps);
    end
end
