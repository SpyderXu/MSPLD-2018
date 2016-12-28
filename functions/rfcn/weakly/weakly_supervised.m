function model_path = weakly_supervised(roidb_train, solver_file, model_file, val_interval, snapshot_interval, box_param, conf, cache_dir, prefix, suffix, final_name, step_epoch, max_epoch)
  model_path = fullfile(cache_dir, [prefix, '_', final_name, suffix]);
  assert(isfield(box_param, 'bbox_means'));
  assert(isfield(box_param, 'bbox_stds'));
  %if (exist(model_path, 'file'))
  %  fprintf('Exist Caffe Model : %s, Skiped\n', model_path);
  %  return;
  %end
  caffe.reset_all();
  caffe_solver = caffe.Solver(solver_file);
  caffe_solver.net.copy_from(model_file);
  total_num = numel(roidb_train);
  caffe_solver.set_max_iter(total_num * max_epoch);
  caffe_solver.set_stepsize(total_num * step_epoch);

  shuffled_inds = [];
  train_results = [];
  max_iter = caffe_solver.max_iter();
  fprintf('********** %6s Training : total[%5d] max_iter[%6d] *************\n', prefix, total_num, max_iter);
  caffe_solver.net.set_phase('train');

  while (caffe_solver.iter() < max_iter)

        [shuffled_inds, sub_db_inds] = weakly_generate_random_minibatch(shuffled_inds, roidb_train, conf.ims_per_batch);
        net_inputs = weakly_get_minibatch(conf, roidb_train(sub_db_inds));

        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);

        rst = caffe_solver.net.get_output();
        train_results = parse_rst(train_results, rst);

        % do valdiation per val_interval iterations
        if mod(caffe_solver.iter(), val_interval) == 0
            weakly_show_state(caffe_solver.iter(), max_iter, train_results);
            train_results = [];
            diary; diary; % flush diary
        end
        % weakly_snapshot
        if mod(caffe_solver.iter(), snapshot_interval) == 0
            iter_model_path = fullfile(cache_dir, [prefix, '_iter_', num2str(caffe_solver.iter()), suffix]);
            weakly_snapshot(caffe_solver, box_param.bbox_means, box_param.bbox_stds, iter_model_path);
        end

    end
    % final weakly_snapshot
    weakly_snapshot(caffe_solver, box_param.bbox_means, box_param.bbox_stds, model_path);

end
