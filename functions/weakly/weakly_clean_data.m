function [image_roidb_train] = weakly_clean_data(classes, image_roidb_train, smallest)

  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num);               begin_time = tic;
  %% Filter Multiple Boxes
  lower_score = cell(num_class,1);
  for idx = 1:num
    pseudo_boxes = check_filter_img(image_roidb_train(idx).pseudo_boxes, smallest);
    if (isempty(pseudo_boxes)), continue; end
    pseudo_boxes = check_save_max(pseudo_boxes, 0.1);
    if (isempty(pseudo_boxes)), continue; end

    image_roidb_train(idx).pseudo_boxes = pseudo_boxes;
    oks(idx) = true;
    for j = 1:numel(pseudo_boxes)
        class = pseudo_boxes(j).class;
        score = pseudo_boxes(j).score;
        lower_score{class}(end+1) = score;
    end
  end

  image_roidb_train = image_roidb_train(oks);
  fprintf('weakly_filter_roidb after filter left %4d images\n', numel(image_roidb_train));

  weakly_debug_info( classes, image_roidb_train );
end

function pseudo_boxes = check_save_max(pseudo_boxes, min_thresh)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  unique_cls = unique(class);

  keep = true(numel(class), 1);
  for c = 1:numel(unique_cls)
    cls = unique_cls(c);
    idx = find(class == cls);
    curkeep = nms_min([boxes(idx,:), score(idx,:)], min_thresh);
    keep( idx(curkeep) ) = true;
  end
  pseudo_boxes = pseudo_boxes(keep);

end

function ok = check_filter_img(pseudo_boxes, smallest)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  boxes = {pseudo_boxes.box};   boxes = cat(1, boxes{:});
  keepB = find(boxes(:,3)-boxes(:,1) >= smallest);
  keepA = find(boxes(:,4)-boxes(:,2) >= smallest);
  keep  = intersect(keepA, keepB);
  pseudo_boxes = pseudo_boxes(keep);
  class = class(keep);
  ok = [];
  if (numel(unique(class)) >= 4), return; end
  for i = 1:numel(class)
    if (numel(find(class == class(i))) >= 4), return; end
  end
  ok = pseudo_boxes;
end
