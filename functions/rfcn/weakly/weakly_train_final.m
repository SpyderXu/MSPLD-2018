function save_model_path = weakly_train_final(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                                @isstruct);
    ip.addRequired('imdb_train',                          @iscell);
    ip.addRequired('roidb_train',                         @iscell);
    ip.addParamValue('max_epoch',         5,              @isscalar); 
    ip.addParamValue('step_epoch',        5,              @isscalar); 
    ip.addParamValue('val_interval',      500,            @isscalar); 
    ip.addParamValue('snapshot_interval', 300,            @isscalar);
    ip.addParamValue('solver_def_file',   'un-define',    @isstr);
    ip.addParamValue('test_def_file',     'un-define',    @isstr);
    ip.addParamValue('net_file',          'un-define',    @isstr);
    ip.addParamValue('cache_name',        'un-define',    @isstr);
    ip.addParamValue('box_param',         struct(),       @isstruct);

    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    assert(isfield(opts, 'box_param'));
    assert(isfield(conf, 'classes'));
    assert(isfield(conf, 'per_class_sample'));
    assert(isfield(conf, 'debug'));
    assert(isfield(conf, 'base_select'));
    assert(isfield(conf, 'nms_config'))
    assert(isfield(opts.box_param, 'bbox_means'));
    assert(isfield(opts.box_param, 'bbox_stds'));
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
    conf.debug_cache_dir =  fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');
    
    if (numel(conf.per_class_sample) == 1) 
        conf.per_class_sample = conf.per_class_sample * ones(numel(conf.classes), 1);
    end
%% init
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);

    % set gpu/cpu
    caffe.reset_all();
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    seed_rand(conf.rng_seed);

%% making tran/val data
    fprintf('Preparing training data...');
    %[image_roidb_train] = score_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);
    [image_roidb_train] = rfcn_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds);
    [warmup_roidb_train, image_roidb_train] = weakly_sample_train(image_roidb_train, conf.per_class_sample, opts.imdb_train{1}.flip);
    %Draw Warmup -- Debug
    weakly_draw_warm(conf, warmup_roidb_train, 'sampled_warmup');
    %weakly_draw_warm(conf,  image_roidb_train, 'unlabeled_train_data');
    fprintf('Done.\n');
    
    %%% Clear gt from image_roidb_train and Filter Over Single Roidb
    count_single_label = 0;
    filtered_image_roidb_train = [];
    for index = 1:numel(image_roidb_train)
        gt = image_roidb_train(index).GT_Index;
        Struct = struct('image_path', image_roidb_train(index).image_path, ...
                        'image_id',   image_roidb_train(index).image_id, ...
                        'imdb_name',  image_roidb_train(index).imdb_name, ...
                        'im_size',    image_roidb_train(index).im_size, ...
                        'overlap',    [], ...
                        'boxes',      image_roidb_train(index).boxes(~gt, :), ...
                        'bbox_targets', [], ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                        'image_label',image_roidb_train(index).image_label);
        filtered_image_roidb_train{end+1} = Struct;
    end
    fprintf('Images after filtered : %d, total : %d\n', numel(filtered_image_roidb_train), numel(image_roidb_train));
    image_roidb_train = cat(1, filtered_image_roidb_train{:});
    %% New Warmup
    filtered_image_roidb_train = [];
    for index = 1:numel(warmup_roidb_train)
        gt = warmup_roidb_train(index).GT_Index;
        Struct = struct('image_path', warmup_roidb_train(index).image_path, ...
                        'image_id',   warmup_roidb_train(index).image_id, ...
                        'imdb_name',  warmup_roidb_train(index).imdb_name, ...
                        'im_size',    warmup_roidb_train(index).im_size, ...
                        'overlap',    warmup_roidb_train(index).overlap, ...
                        'boxes',      warmup_roidb_train(index).boxes, ...
                        'bbox_targets', warmup_roidb_train(index).bbox_targets, ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', warmup_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', warmup_roidb_train(index).boxes(gt, :), ...
                        'image_label',warmup_roidb_train(index).image_label);
        filtered_image_roidb_train{end+1} = Struct;
    end
    warmup_roidb_train = cat(1, filtered_image_roidb_train{:});
    %% Show Box Per class
    num_class = numel(conf.classes);
    boxes_per_class   = zeros(num_class, 2);
    for index = 1:numel(image_roidb_train)
        class = image_roidb_train(index).Debug_GT_Cls;
        for j = 1:numel(class)
            boxes_per_class(class(j), 1) = boxes_per_class(class(j), 1) + 1;
        end
    end
    for index = 1:numel(warmup_roidb_train)
        class = warmup_roidb_train(index).Debug_GT_Cls;
        for j = 1:numel(class)
            boxes_per_class(class(j), 2) = boxes_per_class(class(j), 2) + 1;
        end
    end
    for index = 1:num_class
        fprintf('%13s : unlabeled boxes : %5d,  labeled boxes : %3d\n', conf.classes{index}, boxes_per_class(index, 1), boxes_per_class(index, 2));
    end
    clear class j timestamp log_file index filtered_image_roidb_train;
	boxes_per_class = zeros(num_class, 1);
	for index = 1:numel(warmup_roidb_train)
        class = warmup_roidb_train(index).image_label;
        for j = 1:numel(class)
            boxes_per_class(class(j)) = boxes_per_class(class(j)) + 1;
        end
	end

%% Add conf flip attr
    conf.flip = opts.imdb_train{1}.flip;
    for idx = 1:numel(opts.imdb_train)
        assert (opts.imdb_train{idx}.flip == 1);
    end
    
%% training
    model_suffix  = '.caffemodel';
    
    previous_model = weakly_supervised(warmup_roidb_train, opts.solver_def_file, opts.net_file, opts.val_interval, opts.snapshot_interval, ...
		opts.box_param, conf, cache_dir, 'Loop_0', model_suffix, 'final', opts.step_epoch, opts.max_epoch);

    Init_Per_Select = [40, 10, 4, 5, 2, 4, 30, 13, 15, 4,...
                        4, 10, 11, 7, 30, 7, 7, 10, 20, 10];

    for index = 1:numel(conf.base_select)
        base_select = conf.base_select(index);
        begin__time  = tic;
        fprintf('>>>>>>>>>>>>>>>>> Start Loop %2d with base_select : %4.2f\n', index, base_select);

        %% Generate pseudo label
        caffe.reset_all();
        caffe_test_net = caffe.Net(opts.test_def_file, 'test');
        caffe_test_net.copy_from(previous_model);
        [A_image_roidb_train, ~] = weakly_generate_pseudo(conf, {caffe_test_net}, image_roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds, false);

        PER_Select = ceil(Init_Per_Select / min(Init_Per_Select) * base_select);
        %% Filter Unreliable Image with pseudo-boxes
        %[B_image_roidb_train, ~] = weakly_filter_roidb(conf, {caffe_test_net}, A_image_roidb_train, 15, 0.3);
        [B_image_roidb_train, ~] = weakly_filter_roidb(conf, {caffe_test_net}, A_image_roidb_train, 15, PER_Select*2);

        if (conf.debug), inloop_debug(conf, B_image_roidb_train, ['Loop_', num2str(index), '_B']); end

        %% Self-Paced Sample
        %PER_Select               = boxes_per_class / min(boxes_per_class) * base_select;
        caffe.reset_all();
        train_solver = caffe.Solver(opts.solver_def_file);
        train_solver.net.copy_from(previous_model);
        C_image_roidb_train      = weakly_generate_v(conf, train_solver, B_image_roidb_train, PER_Select);

        if (conf.debug), inloop_debug(conf, C_image_roidb_train, ['Loop_', num2str(index), '_C']); end

        new_image_roidb_train = [warmup_roidb_train; C_image_roidb_train];

        fprintf('.....weakly_supervised train, prepare cost %.1f s ...................\n', toc(begin__time));
        previous_model        = weakly_supervised(new_image_roidb_train, opts.solver_def_file, opts.net_file, opts.val_interval, opts.snapshot_interval, ...
                                                  opts.box_param, conf, cache_dir, ['Loop_', num2str(index)], model_suffix, 'final', opts.step_epoch, opts.max_epoch);

        %%% Check Whether Stop
        %if (numel(B_image_roidb_train) * 0.9 <= sum(PER_Select))
        %    fprintf('Stop iteration due to reach max numbers : %d\n', ceil(numel(B_image_roidb_train) * 0.9));
        %    break;
        %end

    end
    
    weakly_final_model = sprintf('final%s', model_suffix);
    save_model_path    = fullfile(cache_dir, weakly_final_model);
    copyfile(previous_model, save_model_path);

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function inloop_debug(conf, image_roidb_train, debug_dir)
  for iii = 1:numel(image_roidb_train)
    weakly_debug_final(conf, image_roidb_train(iii), debug_dir);
  end
end
