function save_model_path = weakly_train(conf, imdb_train, roidb_train, varargin)
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
    ip.addParamValue('do_val',            false,          @isscalar);
    ip.addParamValue('val_interval',      500,            @isscalar); 
    ip.addParamValue('snapshot_interval', 300,            @isscalar);
    ip.addParamValue('solver_def_file',   'un-define',    @isstr);
    ip.addParamValue('net_file',          'un-define',    @isstr);
    ip.addParamValue('cache_name',        'un-define',    @isstr);
    ip.addParamValue('caffe_version',     'Unkonwn',      @isstr);
    ip.addParamValue('box_param',         struct(),       @isstruct);

    ip.parse(conf, imdb_train, roidb_train, varargin{:});
    opts = ip.Results;
    assert(isfield(opts, 'box_param'));
    assert(isfield(conf, 'classes'));
    assert(isfield(conf, 'SPLD'));
    assert(isfield(conf, 'allow_mul_ins'));
    assert(isfield(conf, 'per_class_sample'));
    %assert(isfield(conf, 'upper_for_min'));
    assert(isfield(conf, 'debug'));
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
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
    disp('conf:');
    disp(conf);
    disp('opts:');
    disp(opts);
    disp(conf.SPLD);
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
    
    %%% Clear gt from image_roidb_train 
    %%and Filter Over Single Roidb
    count_single_label = 0;
    single_image_roidb_train = [];
    for index = 1:numel(image_roidb_train)
        Struct = image_roidb_train(index);
        gt = image_roidb_train(index).GT_Index;
        Struct.overlap      = [];%image_roidb_train(index).overlap(~gt, :);
        Struct.Debug_GT_Cls = image_roidb_train(index).class(gt, :);
        Struct.Debug_GT_Box = image_roidb_train(index).boxes(gt, :);
        Struct.boxes        = image_roidb_train(index).boxes(~gt, :);
        Struct.bbox_targets = [];%image_roidb_train(index).class(~gt, :);
        %image_roidb_train(index) = rmfield(image_roidb_train(index), {'overlap';'class';'gt'});
        %if (numel(image_roidb_train(index).image_label) == 1)
        if (sum(gt) == numel(setdiff(unique(Struct.image_label),[0])) || conf.allow_mul_ins)
            count_single_label = count_single_label + 1;
            single_image_roidb_train{end+1} = Struct;
        end
        %image_roidb_train(index).image_label = [];
    end
    fprintf('Images after filtered : %d, total : %d\n', count_single_label, numel(image_roidb_train));
    image_roidb_train = cat(1, single_image_roidb_train{:});
    %% Show Box Per class
    boxes_per_class   = zeros(numel(conf.classes), 1);
    for index = 1:numel(image_roidb_train)
        class = image_roidb_train(index).Debug_GT_Cls;
        for j = 1:numel(class)
            boxes_per_class(class(j)) = boxes_per_class(class(j)) + 1;
        end
    end
    for index = 1:numel(conf.classes)
        fprintf('%13s : %5d\n', conf.classes{index}, boxes_per_class(index));
    end
    clear boxes_per_class class j timestamp log_file index;
    %% 

%% Add conf flip attr
    conf.flip = opts.imdb_train{1}.flip;
    for idx = 1:numel(opts.imdb_train)
        assert (opts.imdb_train{idx}.flip == 1);
    end

    
%% training
    model_suffix  = '.caffemodel';
    caffe.reset_all();
    caffe_solver = caffe.Solver(opts.solver_def_file);
    caffe_solver.net.copy_from(opts.net_file);
    shuffled_inds = [];
    train_results = [];
    max_iter = caffe_solver.max_iter();

    lambda    = conf.SPLD.lambda;
    gamma     = conf.SPLD.gamma;
    mx_select = conf.mx_select;
    Last_iters = 0;
    fprintf('********** Start Training : total %d epoches *************\n', max_iter);
    while (caffe_solver.iter() < max_iter)
        if (Last_iters == 0)
            shuffled_inds = [];
            train_results = [];
            new_image_roidb_train = weakly_get_fake_gt(conf, caffe_solver, image_roidb_train, ...
                                              opts.box_param.bbox_means, opts.box_param.bbox_stds, lambda, gamma, mx_select);

            new_image_roidb_train = [warmup_roidb_train; warmup_roidb_train; new_image_roidb_train];
            
            fprintf('(%5d) The Last Epoch Done, lambda : %.4f, gamma : %.4f |||| Sample %d training data\n', caffe_solver.iter(), lambda, gamma, numel(new_image_roidb_train));
            Last_iters = numel(new_image_roidb_train);
            assert (Last_iters > 0);
            if (mx_select <= 0), mx_select = mx_select + 1;
            else,                mx_select = mx_select + max(mx_select*0.2, 2); end
        else
            Last_iters = Last_iters - 1;
        end
        caffe_solver.net.set_phase('train');

        % generate minibatch training data
        % generate data asynchronously 
        [shuffled_inds, sub_db_inds] = weakly_generate_random_minibatch(shuffled_inds, new_image_roidb_train, conf.ims_per_batch);
        net_inputs = weakly_get_minibatch(conf, new_image_roidb_train(sub_db_inds));
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        
        rst = caffe_solver.net.get_output();
        train_results = parse_rst(train_results, rst);
            
        % do valdiation per val_interval iterations
        if ~mod(caffe_solver.iter(), opts.val_interval)
            weakly_show_state(caffe_solver.iter(), max_iter, train_results);
            train_results = [];
            diary; diary; % flush diary
        end
        

        % weakly_snapshot
        if ~mod(caffe_solver.iter(), opts.snapshot_interval)
            weakly_snapshot(caffe_solver, opts.box_param.bbox_means, opts.box_param.bbox_stds, fullfile(cache_dir, sprintf('iter_%d%s', caffe_solver.iter(), model_suffix)));
        end
        
    end
    
    % final weakly_snapshot
    %weakly_snapshot(caffe_solver, opts.box_param.bbox_means, opts.box_param.bbox_stds, cache_dir, sprintf('iter_%d', iter_));
    weakly_final_model = sprintf('final%s', model_suffix);
    save_model_path    = fullfile(cache_dir, weakly_final_model);
    weakly_snapshot(caffe_solver, opts.box_param.bbox_means, opts.box_param.bbox_stds, save_model_path);

    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
end
