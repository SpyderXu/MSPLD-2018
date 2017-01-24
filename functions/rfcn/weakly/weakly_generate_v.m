function [new_image_roidb_train] = weakly_generate_v(conf, train_solver, image_roidb_train, PER_Select, LIMIT)

  train_solver.net.set_phase('test'); 
  number = numel(image_roidb_train);
  classes = conf.classes;

  Loss = Inf(numel(image_roidb_train), numel(classes));

  count_per_class = zeros(numel(classes), 1);
  begin_time = tic;
  for idx = 1:number
    if (rem(idx, 500) == 0 || idx == number), fprintf('weakly_generate_v : handle %4d / %4d image_roidb_train, cost %.2f s\n', idx, number, toc); end

    class = {image_roidb_train(idx).pseudo_boxes.class}; 
    class = cat(1, class{:}); class = unique(class);
    count_per_class(class) = count_per_class(class) + 1;

    loss = get_loss(conf, train_solver, image_roidb_train(idx));
    for j = 1:numel(class)
      Loss(idx, class(j)) = loss;
    end
  end

  %SEL_PER_CLS = weakly_cal_sample_num(PER_Select, count_per_class, LIMIT);
  SEL_PER_CLS = PER_Select;

  cur_keep = false(numel(image_roidb_train), 1);
  for cls = 1:numel(classes)
    [mx_score, mx_ids] = sort(Loss(:, cls));
    for j = 1:min(number, SEL_PER_CLS(cls))
      if (mx_score(j) < 1)
        cur_keep( mx_ids(j) ) = true;
      else
        break;
      end
    end
  end
  
  new_image_roidb_train = image_roidb_train(cur_keep);

  final_count = zeros(numel(classes), 1);
  trueo_count = zeros(numel(classes), 1);
  missd_count = zeros(numel(classes), 1);
  total_count = zeros(numel(classes), 1);
  for i = 1:numel(new_image_roidb_train)
    image_label = new_image_roidb_train(i).image_label;
    class = {new_image_roidb_train(i).pseudo_boxes.class};
    class = unique(cat(1, class{:}));
    for j = 1:numel(class)
      final_count(class(j)) = final_count(class(j)) + 1;
      if (find(image_label==class(j)))
        trueo_count(class(j)) = trueo_count(class(j)) + 1;
      end
    end
    class = setdiff(image_label, class);
    for j = 1:numel(class), missd_count(class(j)) = missd_count(class(j)) + 1; end
    for j = 1:numel(image_label), total_count(image_label(j)) = total_count(image_label(j)) + 1; end
  end
  
  for Cls = 1:numel(classes)
    loss = Loss(cur_keep, Cls);
    loss = loss(find(loss~=inf));
    fprintf('[%02d] [%12s] : [count : %3d / should : %3d / select : %3d] [FINAL= (OK) %3d/%3d Mis: %3d/%3d] : Accuracy : %.4f :| loss : [%.2f, %.2f]\n', Cls, classes{Cls}, ...
                 count_per_class(Cls), ceil(PER_Select(Cls)), SEL_PER_CLS(Cls), trueo_count(Cls), final_count(Cls), missd_count(Cls), total_count(Cls), ...
                 trueo_count(Cls) / final_count(Cls), min(loss), max(loss));
  end
  fprintf('weakly_generate_v end : [accuracy: %.3f (%4d/%4d)], [miss: (%.3f,%.3f) (%4d/%4d)] , cost %.1f s\n', sum(trueo_count) / sum(final_count), sum(trueo_count), sum(final_count), ...
            sum(missd_count)/sum(total_count), mean(missd_count./total_count), sum(missd_count), sum(total_count), toc(begin_time));

end

function loss = get_loss(conf, solver, roidb_train)
  net_inputs = weakly_get_minibatch(conf, roidb_train);
  solver.net.reshape_as_input(net_inputs);
  solver.net.set_input_data(net_inputs);
  solver.net.forward(net_inputs);
  rst = solver.net.get_output();
  assert (strcmp(rst(2).blob_name, 'loss_bbox') == 1);
  assert (strcmp(rst(3).blob_name, 'loss_cls') == 1);
  loss = rst(2).data + rst(3).data;
end
