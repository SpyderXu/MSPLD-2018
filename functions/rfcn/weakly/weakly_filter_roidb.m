function [new_image_roidb_train, keep] = weakly_filter_roidb(conf, test_nets, image_roidb_train, smallest, SAVE_TERM)
  classes = conf.classes;
  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num, 2);            begin_time = tic;
  %save_ratio = 0.30;
  multibox_thresh = 0;
  %% Filter Multiple Boxes
  for idx = 1:num
    oks(idx, 1) = check_filter_img(image_roidb_train(idx).pseudo_boxes, smallest);
    if (oks(idx, 1) && false) 
      ok = check_multibox(conf, test_nets, image_roidb_train(idx), multibox_thresh);
      if (all(ok) == false), oks(idx, 1) = false; end
    end
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
    [scorted_score, sorted_idx] = sort(scores, 'descend');
    if (numel(SAVE_TERM) == 1)
        save_num = ceil(numel(scores) * SAVE_TERM);
    else
        assert (numel(SAVE_TERM) == num_class);
        save_num = SAVE_TERM(cls);
    end
    save_num = min(save_num, numel(scores));
    %lower_score = min(scorted_score(save_num), 0.4);
    lower_score = scorted_score(save_num);
    %idxess = idxess(sorted_idx(end-save_num:end));
    %oks(idxess, 2) = true;
    idxess = idxess(scores <= lower_score);
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
  all_t_count = zeros(numel(classes), 1);
  for i = 1:numel(image_roidb_train)
    image_label = image_roidb_train(i).image_label;
    for j = 1:numel(image_label), all_t_count(image_label(j)) = all_t_count(image_label(j)) + 1; end
  end
  for i = 1:numel(new_image_roidb_train)
    image_label = new_image_roidb_train(i).image_label;
    class = {new_image_roidb_train(i).pseudo_boxes.class}; class = cat(1, class{:});
    score = {new_image_roidb_train(i).pseudo_boxes.score}; score = cat(1, score{:});
    %assert(numel(class) == numel(unique(class)));
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

  miss_mean = get_mean(missd_count, total_count);
  accu_mean = get_mean(trueo_count, final_count);

  for Cls = 1:numel(classes)
    fprintf('--[%02d] [%12s] : [Select= (OK) %3d/%3d | mis: %3d/%3d/%4d] : Accuracy : %.4f :| score : [%.2f, %.2f]\n', Cls, classes{Cls}, ...
                 trueo_count(Cls), final_count(Cls), missd_count(Cls), total_count(Cls), all_t_count(Cls), accu_mean(Cls), score_lowet(Cls), score_hight(Cls));
  end
  fprintf('weakly_filter_roidb [total : %4d]->[F1 : %4d]->[F2 : %3d], [accuracy: (%.3f,%.3f) (%4d/%4d)] [miss: (%.3f,%.3f) (%4d/%4d/%4d) cost %.1f s\n', numel(image_roidb_train), sum(oks(:,1)), sum(oks(:,2)), ...
                sum(trueo_count)/sum(final_count), mean(accu_mean), sum(trueo_count), sum(final_count), ...
                sum(missd_count)/sum(total_count), mean(missd_count./total_count), sum(missd_count), sum(total_count), sum(all_t_count), toc(begin_time));
end

function miss_mean = get_mean(missd_count, total_count)
  total_count(total_count==0) = 1;
  miss_mean = missd_count ./ total_count;
end

function ok = check_filter_img(pseudo_boxes, smallest)
  class = {pseudo_boxes.class};
  class = cat(1, class{:});
  boxes = {pseudo_boxes.box};
  boxes = cat(1, boxes{:});
  if (numel(class) > 3)
    ok = false;
  elseif (numel(class) > numel(unique(class))+1)
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

function ok = check_multibox(conf, test_nets, roidb_train, thresh)
  max_rois_num_in_gpu = 10000;
  boxes = {roidb_train.pseudo_boxes.box}; boxes = cat(1, boxes{:});
  cls   = {roidb_train.pseudo_boxes.class}; cls = cat(1, cls{:});
  ori_score = {roidb_train.pseudo_boxes.score}; ori_score = cat(1, ori_score{:});
  num_boxes = size(boxes, 1);
  Fboxes = []; Fscores = [];
  for j = 1:numel(test_nets)
    [Tboxes, Tscores] = weakly_im_detect(conf, test_nets{j}, imread(roidb_train.image_path), multibox(boxes), max_rois_num_in_gpu);
    if (j == 1)
      Fboxes = Tboxes; Fscores = Tscores;
    else
      Fboxes = Fboxes + Tboxes; Fscores = Fscores + Tscores;
    end
  end
  Fboxes = Fboxes / numel(test_nets);
  Fscores = Fscores / numel(test_nets);
  ok = true(num_boxes, 1);
  for idx = 1:4
    cscores = Tscores((idx-1)*num_boxes+1 : idx*num_boxes, :);
    [mx_score, mx_cls] = max(cscores, [], 2);
    for j = 1:numel(cls)
      if(mx_cls(j) == cls(j) && mx_score(j)+thresh > ori_score(j))
        ok(j) = false;
      end
    end
  end
end


function boxes = multibox(boxes)
  ANS = zeros(0, 4);
  CUR = boxes; CUR(:,4) = (CUR(:,2)+CUR(:,4)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,2) = (CUR(:,2)+CUR(:,4)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,3) = (CUR(:,1)+CUR(:,3)) / 2;
  ANS = [ANS;CUR];
  CUR = boxes; CUR(:,1) = (CUR(:,1)+CUR(:,3)) / 2;
  ANS = [ANS;CUR];
  boxes = ANS;
end

