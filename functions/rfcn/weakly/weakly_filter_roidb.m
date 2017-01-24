function [new_image_roidb_train, keep] = weakly_filter_roidb(image_roidb_train, classes, smallest)
  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num, 3);            begin_time = tic;
  save_ratio = 0.30;
  %% Filter Multiple Boxes
  for idx = 1:num
    oks(idx, 1) = check_filter_img(image_roidb_train(idx).pseudo_boxes, smallest);
  end

  %% Filter low value pseudo_boxes
  values = cell(num_class, 1);
  indexs = cell(num_class, 1);
  for idx = 1:num
    if (oks(idx,1) == false), continue; end
    class = {image_roidb_train(idx).pseudo_boxes.class}; class = cat(1, class{:});
    score = {image_roidb_train(idx).pseudo_boxes.score}; score = cat(1, score{:});
    assert (numel(class) == numel(score));
    assert (all(class <= num_class));
    for j = 1:numel(class)
      values{class(j)}(end+1) = score(j);
      indexs{class(j)}(end+1) = idx;
    end
  end
  oks(:,2) = oks(:,1);
  for cls = 1:num_class
    scores = values{cls};
    idxess = indexs{cls};
    if (isempty(scores)), continue; end
    [~, sorted_idx] = sort(scores);
    save_num = ceil(numel(sorted_idx) * save_ratio);
    save_num = min(save_num, numel(sorted_idx));
    %idxess = idxess(sorted_idx(end-save_num:end));
    %oks(idxess, 2) = true;
    idxess = idxess(sorted_idx(1:end-save_num));
    oks(idxess, 2) = false;
  end

  keep = find(oks(:,2));
  new_image_roidb_train = image_roidb_train(keep);

  final_count = zeros(numel(classes), 1);
  trueo_count = zeros(numel(classes), 1);
  missd_count = zeros(numel(classes), 1);
  total_count = zeros(numel(classes), 1);
  score_hight = zeros(numel(classes), 1);
  score_lowet = inf(numel(classes), 1);
  for i = 1:numel(new_image_roidb_train)
    image_label = new_image_roidb_train(i).image_label;
    class = {new_image_roidb_train(i).pseudo_boxes.class}; class = cat(1, class{:});
    score = {new_image_roidb_train(i).pseudo_boxes.score}; score = cat(1, score{:});
    assert(numel(class) == numel(unique(class)));
    for j = 1:numel(class)
      score_hight(class(j)) = max(score_hight(class(j)), score(j));
      score_lowet(class(j)) = min(score_lowet(class(j)), score(j));
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
    fprintf('--[%02d] [%12s] : [Select= (OK) %3d/%3d | mis: %3d/%3d] : Accuracy : %.4f :| score : [%.2f, %.2f]\n', Cls, classes{Cls}, ...
                 trueo_count(Cls), final_count(Cls), missd_count(Cls), total_count(Cls), trueo_count(Cls) / final_count(Cls), score_lowet(Cls), score_hight(Cls));
  end
  fprintf('weakly_filter_roidb [total : %4d]->[F1 : %4d]->[F2 : %3d], [accuracy: %.3f (%4d/%4d)] [miss: (%.3f,%.3f) (%4d/%4d) cost %.1f s\n', numel(image_roidb_train), sum(oks(:,1)), sum(oks(:,2)), ...
                sum(trueo_count)/sum(final_count), sum(trueo_count), sum(final_count), sum(missd_count)/sum(total_count), mean(missd_count./total_count), sum(missd_count), sum(total_count), toc(begin_time));
end

function ok = check_filter_img(pseudo_boxes, smallest)
  class = {pseudo_boxes.class};
  class = cat(1, class{:});
  boxes = {pseudo_boxes.box};
  boxes = cat(1, boxes{:});
  if (numel(class) > numel(unique(class)))
    ok = false;
  elseif (numel(class) > 4)
    ok = false;
  else
    keepB = find(boxes(:,3)-boxes(:,1) >= smallest);
    keepA = find(boxes(:,4)-boxes(:,2) >= smallest);
    keep  = intersect(keepA, keepB);
    if (numel(keep) ~= numel(class))
      ok = false;
    else
      ok = true;
    end
  end
end
