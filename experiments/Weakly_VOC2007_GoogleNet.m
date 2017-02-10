figure('visible', 'off');
clc;
clear mex;
clear is_valid_handle; % to clear init_key
%%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu();
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% model
model.solver_def_file       = fullfile(pwd, 'models', 'rfcn_prototxts', 'GoogleNet_OHEM', 'solver_lr1_3.prototxt');
model.test_net_def_file     = fullfile(pwd, 'models', 'rfcn_prototxts', 'GoogleNet_OHEM', 'test.prototxt');

model.net_file              = fullfile(pwd, 'models', 'pre_trained_models', 'GoogleNet', 'bvlc_googlenet.caffemodel');
model.mean_image            = fullfile(pwd, 'models', 'pre_trained_models', 'GoogleNet', 'mean_image.mat');
model.extra_para            = fullfile(pwd, 'models', 'pre_trained_models', 'box_param.mat');
model.extra_para            = load(model.extra_para);
assert(exist(model.net_file, 'file') ~= 0, 'GoogleNet Pretrain Model Not Found');
assert(exist(model.mean_image, 'file') ~= 0, 'GoogleNet Mean Image Not Found');

% cache name
opts.cache_name             = 'EWSD_GoogleNet';
% config
conf                        = rfcn_config_ohem('image_means', model.mean_image);
conf.classes                = model.extra_para.VOCopts.classes;
conf.per_class_sample       = 3;
conf.per_class_sample       = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
                               3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
box_param.bbox_means        = model.extra_para.bbox_means;
box_param.bbox_stds         = model.extra_para.bbox_stds;
conf.base_select            = [1, 2, 3, 4];
conf.allow_mul_ins          = true;
conf.debug                  = true;
conf.rng_seed               = 5;
max_epoch                   = 10;
step_epoch                  = 9;
if conf.allow_mul_ins,  multiselect_string = '_multi'; 
else,                   multiselect_string = '_single'; end
opts.cache_name             = [opts.cache_name, '_per-', num2str(mean(conf.per_class_sample)), multiselect_string, ...
                                                '_max-epoch-', num2str(max_epoch), '_stepsize-', num2str(step_epoch), ...
                                                '_seed-', num2str(conf.rng_seed)];
% train/test data
fprintf('Loading dataset...')
dataset                     = [];
dataset                     = Dataset.voc2007_trainval_ss(dataset, 'train', conf.use_flipped);
dataset                     = Dataset.voc2007_test_ss(dataset, 'test', false);
fprintf('Done.\n');

fprintf('-------------------- TRAINING --------------------\n');
train_time                  = tic;
opts.rfcn_model             = weakly_train_final(conf, dataset.imdb_train, dataset.roidb_train, ...
                                'solver_def_file',  model.solver_def_file, ...
                                'net_file',         model.net_file, ...
                                'test_def_file',    model.test_net_def_file, ...
                                'max_epoch',        max_epoch, ...
                                'step_epoch',       step_epoch, ...
                                'cache_name',       opts.cache_name, ...
                                'box_param',        box_param);

assert(exist(opts.rfcn_model, 'file') ~= 0, 'not found trained model');
train_time                  = toc(train_time);
config_save_path            = fullfile(pwd, 'output', 'weakly_cachedir', opts.cache_name, 'config.mat');
save(config_save_path, 'conf', 'opts', 'model', '-v7.3');

fprintf('-------------------- TESTING --------------------\n');
test_time                   = tic;
mAP                         = weakly_co_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                                'net_defs',         {model.test_net_def_file}, ...
                                'net_models',       {opts.rfcn_model}, ...
                                'cache_name',       opts.cache_name,...
                                'ignore_cache',     true);
test_time                   = toc(test_time);
fprintf('Training Cost : %.1f s, Test Cost : %.1f s, mAP : %.2f\n', train_time, test_time, mAP);


fprintf('----------------------------------All Test-----------------------------\n');

imdbs_name          = cell2mat(cellfun(@(x) x.name, dataset.imdb_train,'UniformOutput', false));
test_iters          = [];
rfcn_model          = [];
for iter = 0:numel(conf.base_select)
  test_iters{end+1} = ['Loop_', num2str(iter)];
  rfcn_model{end+1} = fullfile(pwd, 'output', 'weakly_cachedir' , opts.cache_name, imdbs_name, [test_iters{end}, '_final.caffemodel']);
  assert(exist(rfcn_model{end}, 'file') ~= 0, 'not found trained model');
end
mAPs                = [];
for index = 1:numel(rfcn_model)
  if index ~= numel(rfcn_model), test_iteration = 1;
  else,                          test_iteration = 2; end
  mAPs(index)       = weakly_co_test(conf, dataset.imdb_test, dataset.roidb_test, ...
                             'net_defs',         {model.test_net_def_file}, ...
                             'net_models',       rfcn_model(index), ...
                             'cache_name',       opts.cache_name,...
                             'test_iteration',   test_iteration,...
                             'log_prefix',       [test_iters{index}, '_'],...
                             'ignore_cache',     true);
end
