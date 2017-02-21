function [new_image_roidb_train] = weakly_generate_pseudo(test_models, image_roidb_train, boost)

    begin_time = tic;
    caffe.reset_all();                                 classes = test_models{1}.conf.classes;
    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(classes);
    thresh_hold = 0.2;

    test_nets   = cell(numel(test_models), 1);
    confs       = cell(numel(test_models), 1);
    for idx = 1:numel(test_models)
        test_nets{idx} = caffe.Net( test_models{idx}.test_net_def_file , 'test' );
        test_nets{idx}.copy_from( test_models{idx}.cur_net_file );
        assert (test_models{idx}.conf.use_flipped); % argument
        confs{idx} = test_models{idx}.conf;
    end

    for index = 1:num_roidb
        assert (isempty(image_roidb_train(index).overlap));
        assert (isempty(image_roidb_train(index).bbox_targets));
    end
    pseudo_boxes = cell(num_roidb, 1);
    for index = 1:(num_roidb/2)
        if (rem(index, 500) == 0 || index == num_roidb/2)
            fprintf('Handle %4d / %4d image_roidb_train, cost : %.1f s\n', index, num_roidb/2, toc(begin_time));
        end
        reverse_idx = index + (num_roidb/2);
        %if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        cur_roidb_train = {image_roidb_train(index), image_roidb_train(reverse_idx)};
        [boxes] = generate_pseudo(confs, test_nets, cur_roidb_train, num_classes, thresh_hold, boost);
        assert (numel(boxes) == 2);
        pseudo_boxes{index}       = boxes{1};
        pseudo_boxes{reverse_idx} = boxes{2};
    end

    new_image_roidb_train = [];
    for index = 1:num_roidb
        if (isempty(pseudo_boxes{index})), continue; end
        new_image_roidb_train{end+1}            = image_roidb_train(index);
        new_image_roidb_train{end}.pseudo_boxes = pseudo_boxes{index};
    end
    new_image_roidb_train = cat(1, new_image_roidb_train{:});
    weakly_debug_info( classes, new_image_roidb_train );
    fprintf('Generate new_image_roidb_train : %4d -> %4d, Cost : %.1f s\n', num_roidb, numel(new_image_roidb_train), toc(begin_time));
    caffe.reset_all();
end

function structs = generate_pseudo(confs, test_nets, image_roidb_train, num_classes, thresh_hold, boost)
  max_rois_num_in_gpu = 10000;
  assert (numel(image_roidb_train) == 1 || numel(image_roidb_train) == 2);
  box_num = size(image_roidb_train{1}.boxes, 1);
  if (numel(image_roidb_train) == 2)
    assert (all(image_roidb_train{1}.im_size == image_roidb_train{2}.im_size));
    reverse_box = image_roidb_train{1}.boxes;
    reverse_box(:, [1,3]) = image_roidb_train{1}.im_size(2) + 1 - reverse_box(:, [3,1]);
    assert (all(all(reverse_box == image_roidb_train{2}.boxes)));
  end

  final_boxes = [];
  final_score = [];
  for idx = 1:numel(test_nets)
    [boxes, scores] = w_im_detect(confs{idx}, test_nets{idx}, imread(image_roidb_train{1}.image_path), image_roidb_train{1}.boxes, max_rois_num_in_gpu, boost);
    if (numel(image_roidb_train) == 2)
      [rev_boxes, rev_scores] = w_im_detect(confs{idx}, test_nets{idx}, imread(image_roidb_train{2}.image_path), image_roidb_train{2}.boxes, max_rois_num_in_gpu, boost);
      rev_boxes(:, [1,3]) = image_roidb_train{1}.im_size(2) + 1 - rev_boxes(:, [3,1]);
      boxes  = (boxes + rev_boxes) / 2;
      scores = (scores+ rev_scores) / 2;
    end

    if (isempty(final_boxes))
      final_boxes = boxes;
      final_score = scores;
    else
      final_boxes = final_boxes + boxes;
      final_score = final_score + scores;
    end
  end
  final_boxes = final_boxes ./ numel(test_nets); 
  final_score = final_score ./ numel(test_nets);

  [MX_per_class, ID_cls] = max(final_score);
  [MX_per_boxes, ID_bbx] = max(final_score, [], 2);
  [mx_score, mx_class]   = max(MX_per_class);

  pos_structs = [];
  rev_structs = [];
  for Cls = 1:num_classes
    aboxes  = [image_roidb_train{1}.boxes, final_score(:,Cls)];
    %aboxes  = [final_boxes(:,(Cls-1)*4+1:Cls*4), final_score(:,Cls)];
    keep           = nms(aboxes, 0.3);
    keep           = keep(ID_bbx(keep)==Cls);
    keep           = keep(final_score(keep, Cls) >= thresh_hold);
    for j = 1:numel(keep)
      pbox = aboxes(keep(j), 1:4);
      Struct = struct('box',   pbox, ...
                      'score', aboxes(keep(j), 5), ...
                      'class', Cls);
      pos_structs{end+1} = Struct;
      if (numel(image_roidb_train) == 2)
        pbox([1,3]) = image_roidb_train{1}.im_size(2) + 1 - pbox([3,1]);
        Struct = struct('box',   pbox, ...
                        'score', aboxes(keep(j), 5), ...
                        'class', Cls);
        rev_structs{end+1} = Struct;
      end
    end
  end
  if (isempty(pos_structs) == false), pos_structs = cat(1, pos_structs{:}); end
  if (isempty(rev_structs) == false), rev_structs = cat(1, rev_structs{:}); end

  if (numel(image_roidb_train) == 2)
    structs = {pos_structs, rev_structs};
  elseif (numel(image_roidb_train) == 1)
    structs = {pos_structs};
  else
    error('error number of roidb train in generate_pseudo');
  end
end

function [boxes, scores] = w_im_detect(conf, test_net, image, in_boxes, max_rois_num_in_gpu, boost)

  [boxes, scores] = weakly_im_detect(conf, test_net, image, in_boxes, max_rois_num_in_gpu);

  if (boost)
    [~, mx_id] = max(scores, [], 2);
    mx_id = (mx_id-1)*4;
    add_boxes = single(zeros(size(in_boxes)));
    parfor box_id = 1:size(in_boxes,1)
        for coor = 1:4, add_boxes(box_id, coor) = boxes(box_id, mx_id(box_id)+coor); end
    end
    in_boxes = (in_boxes + add_boxes) ./ 2;
    [boxes, scores] = weakly_im_detect(conf, test_net, image, in_boxes, max_rois_num_in_gpu);
  end

end
