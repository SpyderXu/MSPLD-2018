function save_model_path = weakly_co_train_final(conf, imdb_train, roidb_train, models, varargin)
% --------------------------------------------------------
% R-FCN implementation
% Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
% Copyright (c) 2016, Jifeng Dai
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb_train',                        @iscell);
    ip.addRequired('roidb_train',                       @iscell);
    ip.addParamValue('max_epoch',         5,            @isscalar);
    ip.addParamValue('step_epoch',        5,            @isscalar);
    ip.addParamValue('val_interval',      500,          @isscalar); 
    ip.addParamValue('cache_name',        'un-define', @isstr);
    ip.addParamValue('box_param',         struct(),     @isstruct);

    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    assert(iscell(models));
    assert(isfield(opts, 'box_param'));
    assert(isfield(conf, 'classes'));
    assert(isfield(conf, 'per_class_sample'));
    assert(isfield(opts.box_param, 'bbox_means'));
    assert(isfield(opts.box_param, 'bbox_stds'));
    assert(isfield(conf, 'debug'));
    assert(isfield(conf, 'pseudo_way'));
    assert(isfield(conf, 'base_select'));
    assert(isfield(conf, 'nms_config'));
    assert(numel(models) == 2);
    for idx = 1:numel(models)
        assert(isfield(models{idx}, 'solver_def_file'));
        assert(isfield(models{idx}, 'test_net_def_file'));
        assert(isfield(models{idx}, 'net_file'));
        assert(isfield(models{idx}, 'name'));
    end
    
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
    

%% making tran/val data
    fprintf('Preparing training data...');
    %[image_roidb_train] = score_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);
    [image_roidb_train] = rfcn_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds);
    [warmup_roidb_train, image_roidb_train] = weakly_sample_train(image_roidb_train, conf.per_class_sample, opts.imdb_train{1}.flip);
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
                        'image_label', image_roidb_train(index).image_label);
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
                        'image_label', warmup_roidb_train(index).image_label);
        filtered_image_roidb_train{end+1} = Struct;
    end
    warmup_roidb_train = cat(1, filtered_image_roidb_train{:});
    %% Show Box Per class
    num_class = numel(conf.classes);
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
    %boxes_per_class = boxes_per_class(:, 2);
%% assert conf flip attr
    conf.flip = opts.imdb_train{1}.flip;
    for idx = 1:numel(opts.imdb_train)
        assert (opts.imdb_train{idx}.flip == 1);
    end

%% training
    model_suffix   = '.caffemodel';
    previous_model = cell(numel(models), 1);
    for idx = 1:numel(models)
        previous_model{idx} = weakly_supervised(warmup_roidb_train, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                                opts.box_param, conf, cache_dir, [models{idx}.name, '_Loop_0'], model_suffix, 'final', opts.step_epoch, opts.max_epoch);
    end

    LIMIT = 2;
    boost = false;

    pre_keep = false(numel(image_roidb_train), 1);

    Init_Per_Select = [40, 10, 10, 10, 15, 10, 40, 13, 15, 10,...
                       15, 15,  3,  8, 10, 15, 10, 10, 35, 25];
%% Start Training
    for index = 1:numel(conf.base_select)
        base_select = conf.base_select(index);

        for idx = 1:numel(models)
            fprintf('\n-------Start Loop %2d == %8s ==with base_select : %4.2f-------\n', index, models{idx}.name, base_select);
            caffe.reset_all();
            use_for_pseudo = [];
            if (strcmp(conf.pseudo_way, 'both')==1 || strcmp(conf.pseudo_way, 'oppo')==1)
                oppo_test_net = caffe.Net(models{3-idx}.test_net_def_file, 'test');
                oppo_test_net.copy_from(previous_model{3-idx});
                use_for_pseudo{end+1} = oppo_test_net;
            end
            if (strcmp(conf.pseudo_way, 'both')==1 || strcmp(conf.pseudo_way, 'self')==1)
                self_test_net = caffe.Net(models{idx}.test_net_def_file, 'test');
                self_test_net.copy_from(previous_model{idx});
                use_for_pseudo{end+1} = self_test_net;
            end
            assert (numel(use_for_pseudo) > 0);
            
            [A_image_roidb_train, A_keep_id] = weakly_generate_pseudo(conf, use_for_pseudo, image_roidb_train, opts.box_param.bbox_means, opts.box_param.bbox_stds, boost);

            PER_Select = ceil(Init_Per_Select * base_select);
            %% Filter Unreliable Image with pseudo-boxes
            %[B_image_roidb_train, B_keep_id] = weakly_filter_roidb(conf, {oppo_test_net,self_test_net}, A_image_roidb_train, 15, 0.3);
            [B_image_roidb_train, B_keep_id] = weakly_filter_roidb(conf, use_for_pseudo, A_image_roidb_train, 15, PER_Select*LIMIT);
            keep_id = A_keep_id(B_keep_id);

            caffe.reset_all();
            oppo_train_solver = caffe.Solver(models{3-idx}.solver_def_file);
            oppo_train_solver.net.copy_from(previous_model{3-idx});
            self_train_solver = caffe.Solver(models{idx}.solver_def_file);
            self_train_solver.net.copy_from(previous_model{idx});

            [C_image_roidb_train, cur_keep] = weakly_generate_co_v(conf, oppo_train_solver, self_train_solver, B_image_roidb_train, keep_id, pre_keep, PER_Select);

            pre_keep = cur_keep;
            %% Draw
            if (conf.debug), inloop_debug(conf, C_image_roidb_train, ['Loop_', models{idx}.name, '_', num2str(index), '_C']); end

            new_image_roidb_train = [warmup_roidb_train; C_image_roidb_train];
            
            previous_model{idx}   = weakly_supervised(new_image_roidb_train, models{idx}.solver_def_file, models{idx}.net_file, opts.val_interval, ...
                                         opts.box_param, conf, cache_dir, [models{idx}.name, '_Loop_', num2str(index)], model_suffix, 'final', opts.step_epoch, opts.max_epoch);
        end

        %%% Check Whether Stop
        %Uplimit = 0.8;
        %if (numel(B_image_roidb_train) * Uplimit <= sum(PER_Select)) 
        %    fprintf('Stop iteration due to reach max numbers : %d\n', ceil(numel(B_image_roidb_train) * Uplimit));
        %    break;
        %end
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

function inloop_debug(conf, image_roidb_train, debug_dir)
  for iii = 1:numel(image_roidb_train)
    weakly_debug_final(conf, image_roidb_train(iii), debug_dir);
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
