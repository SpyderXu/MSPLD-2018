function save_model_path = weakly_train_v3(conf, imdb_train, roidb_train, varargin)
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
    assert(isfield(conf, 'allow_mul_ins'));
    assert(isfield(conf, 'per_class_sample'));
    assert(isfield(conf, 'debug'));
    assert(isfield(conf, 'base_select'));
    assert(isfield(opts.box_param, 'bbox_means'));
    assert(isfield(opts.box_param, 'bbox_stds'));
    
%% try to find trained model
    imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
    conf.debug_cache_dir =  fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');
    %save_model_path = fullfile(cache_dir, 'final');
    %if exist(save_model_path, 'file')
    %    fprintf('Train : %s Exits, ignore training', save_model_path);
    %    return;
    %end
    
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
        Struct = image_roidb_train(index);
        gt = image_roidb_train(index).GT_Index;
        Struct.overlap      = [];%image_roidb_train(index).overlap(~gt, :);
        Struct.Debug_GT_Cls = image_roidb_train(index).class(gt, :);
        Struct.Debug_GT_Box = image_roidb_train(index).boxes(gt, :);
        Struct.boxes        = image_roidb_train(index).boxes(~gt, :);
        Struct.bbox_targets = [];%image_roidb_train(index).class(~gt, :);
        if (sum(gt) == numel(Struct.image_label) || conf.allow_mul_ins)
            filtered_image_roidb_train{end+1} = Struct;
        end
    end
    fprintf('Images after filtered : %d, total : %d\n', numel(filtered_image_roidb_train), numel(image_roidb_train));
    image_roidb_train = cat(1, filtered_image_roidb_train{:});
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
		class = warmup_roidb_train(index).class;
		class = class(warmup_roidb_train(index).GT_Index);
        for j = 1:numel(class)
            boxes_per_class(class(j), 2) = boxes_per_class(class(j), 2) + 1;
        end
	end
    for index = 1:num_class
        fprintf('%13s : unlabeled boxes : %5d,  labeled boxes : %3d\n', conf.classes{index}, boxes_per_class(index, 1), boxes_per_class(index, 2));
    end
    clear class j timestamp log_file index;
	boxes_per_class = boxes_per_class(:, 2);
%% Add conf flip attr
    conf.flip = opts.imdb_train{1}.flip;
    for idx = 1:numel(opts.imdb_train)
        assert (opts.imdb_train{idx}.flip == 1);
    end
    
%% training
    model_suffix  = '.caffemodel';
    
    previous_model = weakly_supervised(warmup_roidb_train, opts.solver_def_file, opts.net_file, opts.val_interval, opts.snapshot_interval, ...
		opts.box_param, conf, cache_dir, 'Loop_0', model_suffix, 'final', opts.step_epoch, opts.max_epoch);

    for index = 1:numel(conf.base_select)
        base_select = conf.base_select(index);
		caffe.reset_all();
		caffe_test_net = caffe.Net(opts.test_def_file, 'test');
		caffe_test_net.copy_from(previous_model);

        fprintf('>>>>>>>>>>>>>>>>> Start Loop %2d with base_select : %4.2f\n', index, base_select);
		PER_Select            = boxes_per_class / min(boxes_per_class) * base_select;

		debug_dir             = ['Loop_', num2str(index)];
		new_image_roidb_train = weakly_get_fake_gt_v3(conf, caffe_test_net, image_roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds, PER_Select, debug_dir);
		previous_model        = weakly_supervised([warmup_roidb_train;new_image_roidb_train], opts.solver_def_file, opts.net_file, opts.val_interval, opts.snapshot_interval, ...
									opts.box_param, conf, cache_dir, ['Loop_', num2str(index)], model_suffix, 'final', opts.step_epoch, opts.max_epoch);
    end
    
    % final weakly_snapshot
    %weakly_snapshot(caffe_solver, opts.box_param.bbox_means, opts.box_param.bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    weakly_final_model = sprintf('final%s', model_suffix);
    save_model_path    = fullfile(cache_dir, weakly_final_model);
	copyfile(previous_model, save_model_path);

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end
