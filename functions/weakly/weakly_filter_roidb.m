function [new_image_roidb_train] = weakly_filter_roidb(test_models, image_roidb_train, smallest, SAVE_TERM)

  classes = test_models{1}.conf.classes;
  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num, 2);            begin_time = tic;
  %save_ratio = 0.30;
  multibox_thresh = 0;
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

  new_image_roidb_train = image_roidb_train(oks(:,2));
  weakly_debug_info( classes, new_image_roidb_train );
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

