function [image_roidb_train] = weakly_filter_roidb(test_models, image_roidb_train, smallest, SAVE_TERM)

  classes = test_models{1}.conf.classes;
  num = numel(image_roidb_train); num_class = numel(classes);
  oks = false(num);               begin_time = tic;
  %% multibox_thresh = 0;
  %% Filter Multiple Boxes
  lower_score = cell(num_class,1);
  for idx = 1:num
    pseudo_boxes = check_filter_img(image_roidb_train(idx).pseudo_boxes, smallest);
    if (isempty(pseudo_boxes)), continue; end
    pseudo_boxes = check_save_max(pseudo_boxes);
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

  for cls = 1:num_class
    scores = lower_score{cls};
    if (isempty(scores)), continue; end
    [scorted_score, ~] = sort(scores, 'descend');
    lower_score{cls} = scorted_score(min(end, SAVE_TERM(cls)));
  end
  lower_score = cat(1, lower_score{:});

  oks = false(numel(image_roidb_train), 1);
  for idx = 1:numel(image_roidb_train)
    pseudo_boxes = check_filter_score(image_roidb_train(idx).pseudo_boxes, lower_score);
    if (isempty(pseudo_boxes)), continue; end
    
    oks(idx) = true;
    image_roidb_train(idx).pseudo_boxes = pseudo_boxes;
  end

  image_roidb_train = image_roidb_train(oks);
  weakly_debug_info( classes, image_roidb_train );
end

function pseudo_boxes = check_filter_score(pseudo_boxes, lower_score)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  keep = false(numel(pseudo_boxes), 1);
  for i = 1:numel(class)
    if (score(i) >= lower_score( class(i) ))
      keep(i) = true;
    end
  end
  pseudo_boxes = pseudo_boxes(keep);
end

function pseudo_boxes = check_save_max(pseudo_boxes)
  class = {pseudo_boxes.class}; class = cat(1, class{:});
  score = {pseudo_boxes.score}; score = cat(1, score{:});
  unique_cls = unique(class);

  keep = [];
  for j = 1:numel(unique_cls)
    cls = unique_cls(j);
    if (sum(class==cls) >= 4), pseudo_boxes=[]; return; end
    idx = find(class == cls);
    [~, iii] = max(score(idx));
    keep(end+1) = idx(iii);
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
  for i = 1:numel(class)
    if (numel(find(class == class(i))) >=4), return; end
  end
  ok = pseudo_boxes;
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

