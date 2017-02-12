function [new_image_roidb_train, keep] = weakly_generate_pseudo(conf, test_nets, image_roidb_train, bbox_means, bbox_stds, boost)
    assert (conf.flip); % argument

    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(conf.classes);
    tic;
    thresh_hold = 0.2;

    for index = 1:num_roidb
        assert (isempty(image_roidb_train(index).overlap));
        assert (isempty(image_roidb_train(index).bbox_targets));
    end
    pseudo_boxes = cell(num_roidb, 1);
    for index = 1:(num_roidb/2)
        if (rem(index, 500) == 0 || index == num_roidb/2)
            fprintf('Handle %3d / %3d image_roidb_train, cost : %.1f s\n', index, num_roidb/2, toc);
        end
        reverse_idx = index + (num_roidb/2);
        %if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        cur_roidb_train = {image_roidb_train(index), image_roidb_train(reverse_idx)};
        [boxes] = generate_pseudo(conf, test_nets, cur_roidb_train, num_classes, thresh_hold, boost);
        assert (numel(boxes) == 2);
        pseudo_boxes{index}       = boxes{1};
        pseudo_boxes{reverse_idx} = boxes{2};
    end

    new_image_roidb_train = [];
    keep = [];
    for index = 1:num_roidb
        if (isempty(pseudo_boxes{index})), continue; end
        pos_boxes = {pseudo_boxes{index}.box};   pos_boxes = cat(1, pos_boxes{:});
        pos_class = {pseudo_boxes{index}.class}; pos_class = cat(1, pos_class{:});
        pos_score = {pseudo_boxes{index}.score}; pos_score = cat(1, pos_score{:});

        rois        = image_roidb_train(index).boxes;
        rois        = [pos_boxes; rois];
        num_boxes   = size(rois, 1);
        overlap     = zeros(num_boxes, num_classes, 'single');
        for bid = 1:numel(pos_class)
            gt_classes = pos_class(bid);
            gt_boxes   = pos_boxes(bid, :);
            overlap(:, gt_classes) = max(overlap(:, gt_classes), boxoverlap(rois, gt_boxes));
        end
        %append_bbox_regression_targets
        [bbox_targets, is_valid] = weakly_compute_targets(conf, rois, overlap);
        if(is_valid == false), continue; end
        new_image_roidb_train{end+1}            = image_roidb_train(index);
        new_image_roidb_train{end}.boxes        = rois;
        new_image_roidb_train{end}.overlap      = overlap;
        new_image_roidb_train{end}.bbox_targets = bbox_targets;
        new_image_roidb_train{end}.pseudo_boxes = pseudo_boxes{index};
        keep(end+1) = index;
    end

    new_image_roidb_train = cat(1, new_image_roidb_train{:});
    %% Normalize targets
    num_images = length(new_image_roidb_train);
    % Infer number of classes from the number of columns in gt_overlaps
    if conf.bbox_class_agnostic
        num_classes = 1;
    else
        num_classes = size(new_image_roidb_train(1).overlap, 2);
    end
    for idx = 1:num_images
        targets = new_image_roidb_train(idx).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), bbox_means(cls+1, :));
                new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, new_image_roidb_train(idx).bbox_targets(cls_inds, 2:end), bbox_stds(cls+1, :));
            end
        end
    end
    fprintf('Generate new_image_roidb_train : %d, Cost : %.1f s\n', num_images, toc);
end

function structs = generate_pseudo(conf, test_nets, image_roidb_train, num_classes, thresh_hold, boost)
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
    [boxes, scores] = w_im_detect(conf, test_nets{idx}, imread(image_roidb_train{1}.image_path), image_roidb_train{1}.boxes, max_rois_num_in_gpu, boost);
    if (numel(image_roidb_train) == 2)
      [rev_boxes, rev_scores] = w_im_detect(conf, test_nets{idx}, imread(image_roidb_train{2}.image_path), image_roidb_train{2}.boxes, max_rois_num_in_gpu, boost);
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
    if (conf.nms_config == false)
        keep           = nms(aboxes, 0.3);
    else
        keep           = nms_min(aboxes, 0.3);
    end
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
