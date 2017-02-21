function save_model_path = weakly_co_train_final(imdb_train, roidb_train, models, varargin)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    %ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb_train',                        @iscell);
    ip.addRequired('roidb_train',                       @iscell);
    ip.addParamValue('per_class_sample',    3,          @isscalar);
    ip.addParamValue('val_interval',      500,          @isscalar); 
    ip.addParamValue('base_select',       [1],          @isvector); 
    ip.addParamValue('rng_seed',            5,          @isscalar); 
    ip.addParamValue('gamma',             0.3,          @isscalar); 
    ip.addParamValue('debug',            true,          @islogical); 
    ip.addParamValue('use_flipped',      true,          @islogical); 
    ip.addParamValue('boost',           false,          @islogical); 
    ip.addParamValue('cache_name',        'un-define',  @isstr);
    ip.addParamValue('box_param',         struct(),     @isstruct);

    ip.parse(imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    assert(iscell(models));
    assert(isfield(opts, 'box_param'));
    assert(isfield(opts.box_param, 'bbox_means'));
    assert(isfield(opts.box_param, 'bbox_stds'));
    %assert(numel(models) == 2);
    for idx = 1:numel(models)
        assert(isfield(models{idx}, 'solver_def_file'));
        assert(isfield(models{idx}, 'test_net_def_file'));
        assert(isfield(models{idx}, 'net_file'));
        assert(isfield(models{idx}, 'name'));
        assert(isfield(models{idx}, 'conf'));
        assert(isfield(models{idx}.conf, 'classes'));
        assert(isfield(models{idx}.conf, 'max_epoch'));
        assert(isfield(models{idx}.conf, 'step_epoch'));
        assert(isfield(models{idx}.conf, 'regression'));
    end
    
%% try to find trained model
    imdbs_name      = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
    cache_dir       = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, imdbs_name);
    debug_cache_dir = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'debug');
    classes = models{1}.conf.classes;
    if (numel(opts.per_class_sample) == 1)
        opts.per_class_sample = opts.per_class_sample * ones(numel(classes), 1);
    end
%% init
    % set random seed
    prev_rng = seed_rand(opts.rng_seed);
    caffe.set_random_seed(opts.rng_seed);
    
    % init caffe solver
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['co_train_', timestamp, '.txt']);
    diary(log_file);

    % set gpu mode, mush run on gpu
    caffe.reset_all();
    caffe.set_mode_gpu(); 
    
    disp('opts:');
    disp(opts);
    for idx = 1:numel(models)
      fprintf('conf: %2d : %s', idx, models{idx}.name);
      disp(models{idx}.conf);
    end
    

%% making tran/val data
    fprintf('Preparing training data...');
    [image_roidb_train] = weakly_prepare_image_roidb(models{1}.conf, opts.imdb_train, opts.roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds);
    [warmup_roidb_train, image_roidb_train] = weakly_sample_train(image_roidb_train, opts.per_class_sample, opts.imdb_train{1}.flip);
    %Draw Warmup -- Debug
    %weakly_draw_warm(conf, warmup_roidb_train, 'sampled_warmup');
    fprintf('Done.\n');
    
    %%% Clear gt from image_roidb_train and Filter Over Single Roidb
    count_single_label = 0;
    filtered_image_roidb_train = [];
    for index = 1:numel(image_roidb_train)
        gt = image_roidb_train(index).GT_Index;
        Struct = struct('image_path',  image_roidb_train(index).image_path, ...
                        'image_id',    image_roidb_train(index).image_id, ...
                        'imdb_name',   image_roidb_train(index).imdb_name, ...
                        'im_size',     image_roidb_train(index).im_size, ...
                        'overlap',     [], ...
                        'boxes',       image_roidb_train(index).boxes(~gt, :), ...
                        'bbox_targets', [], ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', image_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', image_roidb_train(index).boxes(gt, :), ...
                        'image_label', image_roidb_train(index).image_label, ...
                        'index', index);
        filtered_image_roidb_train{end+1} = Struct;
    end
    fprintf('Images after filtered : %d, total : %d\n', numel(filtered_image_roidb_train), numel(image_roidb_train));
    image_roidb_train = cat(1, filtered_image_roidb_train{:});
    %% New Warmup
    filtered_image_roidb_train = [];
    for index = 1:numel(warmup_roidb_train)
        gt = warmup_roidb_train(index).GT_Index;
        Struct = struct('image_path',  warmup_roidb_train(index).image_path, ...
                        'image_id',    warmup_roidb_train(index).image_id, ...
                        'imdb_name',   warmup_roidb_train(index).imdb_name, ...
                        'im_size',     warmup_roidb_train(index).im_size, ...
                        'overlap',     warmup_roidb_train(index).overlap, ...
                        'boxes',       warmup_roidb_train(index).boxes, ...
                        'bbox_targets', warmup_roidb_train(index).bbox_targets, ...
                        'pseudo_boxes', [], ...
                        'Debug_GT_Cls', warmup_roidb_train(index).class(gt, :), ...
                        'Debug_GT_Box', warmup_roidb_train(index).boxes(gt, :), ...
                        'image_label', warmup_roidb_train(index).image_label, ...
                        'index', index);
        filtered_image_roidb_train{end+1} = Struct;
    end
    warmup_roidb_train = cat(1, filtered_image_roidb_train{:});
    %% Show Box Per class
    num_class = numel(classes);
    boxes_per_class   = zeros(num_class, 2);
    for index = 1:numel(image_roidb_train)
        class = image_roidb_train(index).image_label;
        for j = 1:numel(class)
            boxes_per_class(class(j), 1) = boxes_per_class(class(j), 1) + 1;
        end
    end
    for index = 1:numel(warmup_roidb_train)
        class = warmup_roidb_train(index).image_label;
        for j = 1:numel(class)
            boxes_per_class(class(j), 2) = boxes_per_class(class(j), 2) + 1;
        end
    end
    clear class j timestamp log_file index filtered_image_roidb_train roidb_train imdb_train;
%% assert conf flip attr
    for idx = 1:numel(opts.imdb_train)
        assert ( opts.imdb_train{idx}.flip == opts.use_flipped);
    end

%% training
    model_suffix   = '.caffemodel';
    previous_model = cell(numel(models), 1);
    for idx = 1:numel(models)
        train_mode = weakly_train_mode (models{idx}.conf);
        previous_model{idx} = weakly_supervised(train_mode, warmup_roidb_train, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                                opts.box_param, models{idx}.conf, cache_dir, [models{idx}.name, '_Loop_0'], model_suffix, 'final');

        models{idx}.cur_net_file = previous_model{idx};
    end

    pre_keep = false(numel(image_roidb_train), 1);

    Init_Per_Select = [40, 10, 10, 10, 15, 10, 40, 13, 15, 10,...
                       15, 15,  3,  8, 10, 15, 10, 10, 35, 25];
%% Start Training
    for index = 1:numel(opts.base_select)

        base_select = opts.base_select(index);

        for idx = 1:numel(models)
            fprintf('\n-------Start Loop %2d == %8s ==with base_select : %4.2f-------\n', index, models{idx}.name, base_select);
            
            [A_image_roidb_train] = weakly_generate_pseudo(models, image_roidb_train, opts.boost);

            PER_Select = ceil(Init_Per_Select * base_select);
            %% Filter Unreliable Image with pseudo-boxes
            [B_image_roidb_train] = weakly_filter_roidb(models, A_image_roidb_train, 15, PER_Select);
            if (opts.debug), inloop_debug(B_image_roidb_train, classes, debug_cache_dir, ['L_', num2str(index), '_', models{idx}.name, '_B']); end

            [B_image_roidb_train] = weakly_full_targets(models{idx}.conf, B_image_roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds);
            [C_image_roidb_train] = weakly_generate_co_v(models{idx}, B_image_roidb_train, pre_keep, PER_Select, opts.gamma);
            pre_keep = false(numel(image_roidb_train), 1);
            for j = 1:numel(C_image_roidb_train) , pre_keep(C_image_roidb_train(j).index) = true; end
            if (opts.debug), inloop_debug(C_image_roidb_train, classes, debug_cache_dir, ['L_', num2str(index), '_', models{idx}.name, '_C']); end

            new_image_roidb_train = [warmup_roidb_train; C_image_roidb_train];
            
            train_mode = weakly_train_mode (models{idx}.conf);
            previous_model{idx}   = weakly_supervised(train_mode, new_image_roidb_train, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                                      opts.box_param, models{idx}.conf, cache_dir, [models{idx}.name, '_Loop_', num2str(index)], model_suffix, 'final');
        end

        for idx = 1:numel(models)
            models{idx}.cur_net_file = previous_model{idx};
        end
    end

    save_model_path    = cell(numel(models), 1);
    for idx = 1:numel(models)
        weakly_final_model   = sprintf('%s_final%s', models{idx}.name, model_suffix);
        weakly_final_model   = fullfile(cache_dir, weakly_final_model);
        fprintf('Final [%s] Model Path : %s\n', models{idx}.name, weakly_final_model);
        save_model_path{idx} = weakly_final_model;
        copyfile(previous_model{idx}, save_model_path{idx});
    end

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end

function inloop_debug(image_roidb_train, classes, debug_cache_dir, dir_name)
  debug_cache_dir = fullfile(debug_cache_dir, dir_name);
  for iii = 1:numel(image_roidb_train)
    weakly_debug_final(classes, debug_cache_dir, image_roidb_train(iii));
  end
end

function select = inloop_count(conf, image_roidb_train)
  numcls = numel(conf.classes);
  select = zeros(numcls, 1);
  for index = 1:numel(image_roidb_train)
    class = image_roidb_train(index).image_label;
    for j = 1:numel(class)
      select(class(j)) = select(class(j)) + 1;
    end
  end
end
