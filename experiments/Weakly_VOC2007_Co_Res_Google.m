clc;
clear mex;
clear is_valid_handle; % to clear init_key
%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% model
models                      = cell(2,1);
models{1}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'solver_lr1_3.prototxt');
models{1}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'ResNet-50L_OHEM_res3a', 'test.prototxt');
models{1}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'ResNet-50-model.caffemodel');
models{1}.name              = 'ResNet-50';
assert(exist(models{1}.net_file, 'file') ~= 0, [models{1}.name ' Pretrain Model Not Found']);

models{2}.solver_def_file   = fullfile(pwd, 'models', 'rfcn_prototxts', 'GoogleNet_OHEM', 'solver_lr1_3.prototxt');
models{2}.test_net_def_file = fullfile(pwd, 'models', 'rfcn_prototxts', 'GoogleNet_OHEM', 'test.prototxt');
models{2}.net_file          = fullfile(pwd, 'models', 'pre_trained_models', 'GoogleNet', 'bvlc_googlenet.caffemodel');
models{2}.name              = 'GoogleNet';
assert(exist(models{2}.net_file, 'file') ~= 0, [models{2}.name ' Pretrain Model Not Found']);

% cache name
opts.cache_name             = 'EWSD_Co_Res50_Google';
mean_image                  = fullfile(pwd, 'models', 'pre_trained_models', 'ResNet-50L', 'mean_image.mat');
assert(exist(mean_image, 'file') ~= 0, 'ImageNet Mean Image Not Found');
extra_para                  = load(fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat'));
% config
conf                        = rfcn_config_ohem('image_means', mean_image);
conf.classes                = extra_para.VOCopts.classes;
conf.per_class_sample       = 3;
box_param.bbox_means        = extra_para.bbox_means;
box_param.bbox_stds         = extra_para.bbox_stds;
conf.base_select            = [1, 1.5, 2.2, 3.0, 4];
conf.debug                  = true;
conf.rng_seed               = 5;
max_epoch                   = 10;
step_epoch                  = 9;
opts.cache_name             = [opts.cache_name, '_per-', num2str(conf.per_class_sample), ...
                                                '_max_epoch', num2str(max_epoch), '_stepsize-', num2str(step_epoch), ...
                                                '_seed-', num2str(conf.rng_seed)];
% train/test data
fprintf('Loading dataset...');
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', conf.use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);
fprintf('Done.\n');

fprintf('-------------------- TRAINING --------------------\n');
train_time                  = tic;
opts.rfcn_model             = weakly_co_train_final(conf, dataset.imdb_train, dataset.roidb_train, models, ...
                                'cache_name',       opts.cache_name, ...
                                'max_epoch',        max_epoch, ...
                                'step_epoch',       step_epoch, ...
                                'box_param',        box_param);
assert(isfield(opts, 'rfcn_model') ~= 0, 'not found trained model');
train_time                  = toc(train_time);

fprintf('-------------------- TESTING --------------------\n');
assert(numel(opts.rfcn_model) == numel(models));
mAPs                        = cell(numel(models), 1);
test_time                   = tic;
net_defs                    = [];
net_models                  = [];
for idx = 1:numel(models)
            mAPs{idx} = weakly_co_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {models{idx}.test_net_def_file}, ...
                                'net_models',       opts.rfcn_model(idx), ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       [models{idx}.name, '_final_'],...
                                'ignore_cache',     true);
            net_defs{idx} = models{idx}.test_net_def_file;
            net_models{idx} = opts.rfcn_model{idx};
end
mAPs{3}                     = weakly_co_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         net_defs, ...
                                'net_models',       net_models, ...
                                'cache_name',       opts.cache_name,...
                                'log_prefix',       ['Co_final_'],...
                                'ignore_cache',     true);

test_time                   = toc(test_time);
fprintf('Training Cost : %.1f s, Test Cost : %.1f s\n', train_time, test_time);
for idx = 1:numel(models)
    fprintf('%s mAP : %.3f\n', models{idx}.name, mAPs{idx});
end

fprintf('----------------------------------All Test-----------------------------\n');
rfcn_model          = cell(numel(models), numel(conf.base_select)+1);
for iter = 0:numel(conf.base_select)
  for idx = 1:numel(models)
    rfcn_model{idx, iter+1} = fullfile(pwd, 'output', 'weakly_cachedir' , opts.cache_name, 'voc_2007_trainval', [models{idx}.name, '_Loop_', num2str(iter), '_final.caffemodel']);
	assert(exist(rfcn_model{idx, iter+1}, 'file') ~= 0, 'not found trained model');
  end
end
S_mAPs              = zeros(numel(models), size(rfcn_model,2));
for index = 1:size(rfcn_model, 2)
  for idx = 1:numel(models)
    S_mAPs(idx, index) = weakly_co_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                             'net_defs',         {models{idx}.test_net_def_file}, ...
                             'net_models',       rfcn_model(idx,index), ...
                             'test_iteration',   1, ...
                             'cache_name',       opts.cache_name, ...
                             'log_prefix',       [models{idx}.name, '_', num2str(index-1), '_'], ...
                             'ignore_cache',     true);
  end
end
