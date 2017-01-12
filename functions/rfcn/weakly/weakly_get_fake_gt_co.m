function new_image_roidb_train = weakly_get_fake_gt_co(conf, oppo_test_net, self_test_net, image_roidb_train, bbox_means, bbox_stds, PER_Select, debug_dir)
    assert (conf.flip); % argument

    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(conf.classes);   
    tic;
    thresh_hold = 0.2;
    oppo_all_Select = cell(num_roidb/2, 1);
    self_all_Select = cell(num_roidb/2, 1);
    for index = 1:(num_roidb/2)
        if (rem(index, 500) == 0 || index == num_roidb/2)
            fprintf('Handle %3d / %3d image_roidb_train, cost : %.1f s\n', index, num_roidb/2, toc);
        end
        reverse_idx = index + (num_roidb/2);
        %if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        cur_roidb_train = [image_roidb_train(index), image_roidb_train(reverse_idx)];
        cur_img_ids     = [index, reverse_idx];
        [structs] = weakly_extract_struct_v2(conf, oppo_test_net, cur_roidb_train, cur_img_ids, num_classes, thresh_hold);
        %assert (numel(structs) == num_classes);
        oppo_all_Select{index} = structs;

        [structs] = weakly_extract_struct_v2(conf, self_test_net, cur_roidb_train, cur_img_ids, num_classes, thresh_hold);
        %assert (numel(structs) == num_classes);
        self_all_Select{index} = structs;
    end
    oppo_all_Select = cat(1, oppo_all_Select{:});
    self_all_Select = cat(1, self_all_Select{:});
    
    oppo_all_Select = convert2image( oppo_all_Select, num_roidb ); %% boxes -> images
    self_all_Select = convert2image( self_all_Select, num_roidb );

    oppo_all_Select = weakly_filter_img( oppo_all_Select, conf.classes, 20 );
    self_all_Select = weakly_filter_img( self_all_Select, conf.classes, 20 );

    fprintf('Base filter cost %.2f s\n', toc);
    tic;
    assert(numel(oppo_all_Select) == numel(self_all_Select));
    assert(numel(oppo_all_Select) == num_roidb);
    for idx = 1:num_roidb
        image_path = image_roidb_train(idx).image_path;
        structes = oppo_all_Select{idx};
        if (numel(structes) > 0)
            ok = weakly_filter_img_multibox(conf, oppo_test_net, image_path, structes);
            if (all(ok) == false), oppo_all_Select{idx} = []; end
        end
        structes = self_all_Select{idx};
        if (numel(structes) > 0)
            ok = weakly_filter_img_multibox(conf, self_test_net, image_path, structes);
            if (all(ok) == false), self_all_Select{idx} = []; end
        end
		if (rem(idx, 1000) == 0 || idx == num_roidb), fprintf('Handle %4d / %4d, cost %.1f s \n', idx, num_roidb, toc); end
    end

    %% Select for oppo
    select_per_class = cell(num_classes, 1);
    OPPO_COUNT_PER_CLS = zeros(num_classes, 1);
	for idx = 1:num_roidb
        structs = oppo_all_Select{idx};
        for j = 1:numel(structs)
            cstruct = structs(j);
            select_per_class{cstruct.cls}(end+1) = cstruct;
        end
    end
	for cls = 1:num_classes
	  OPPO_COUNT_PER_CLS(cls) = numel(select_per_class{cls});
    end
    %% Calculate PerMax
    OPPO_SEL_PER_CLS = weakly_cal_sample_num(PER_Select, OPPO_COUNT_PER_CLS, 10);

    %% Merge Into Self Pool
    for Cls = 1:num_classes
        if (numel(select_per_class{Cls}) == 0), continue; end
        temp   = select_per_class{Cls};
        scores = {temp.score}; scores = cat(1, scores{:}); [scores, idx] = sort(scores, 'descend');
        sorted_ins = temp(idx);
        for j = 1:min(OPPO_SEL_PER_CLS(Cls), numel(sorted_ins))
            cstruct = sorted_ins(j);
            self_all_Select{cstruct.img_id} = merge_boxes(self_all_Select{cstruct.img_id}, cstruct);
        end
    end
    select_per_class = cell(num_classes, 1);
    for idx = 1:num_roidb
        structs = self_all_Select{idx};
        for j = 1:numel(structs)
            cstruct = structs(j);
            select_per_class{cstruct.cls}(end+1) = cstruct;
        end
    end
    SELF_COUNT_PER_CLS = zeros(num_classes, 1);
    for cls = 1:num_classes
        SELF_COUNT_PER_CLS(cls) = numel(select_per_class{cls});
    end
    SELF_SEL_PER_CLS = weakly_cal_sample_num(PER_Select, SELF_COUNT_PER_CLS, 10);

    final_per_class = cell(num_classes, 1);
    %ratio = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.7, 0.0, 0.0, ...
    %         0.0, 0.7, 0.7, 0.0, 0.7, 0.0, 0.0, 0.7, 0.7, 0.0];
    ratio = zeros(num_classes, 1);
    sel_for_class   = zeros(num_classes, 1);
    match_for_class = zeros(num_classes, 1);
    lowest_score    = zeros(num_classes, 1) + inf;
    highst_score    = zeros(num_classes, 1);
    for Cls = 1:num_classes
        if (numel(select_per_class{Cls}) == 0), continue; end
        temp   = select_per_class{Cls};
        scores = {temp.score}; scores = cat(1, scores{:}); [scores, idx] = sort(scores, 'descend');
        sorted_ins = temp(idx);
        top_score = scores(1);
        for j = 1:min(SELF_SEL_PER_CLS(Cls), numel(sorted_ins))
            cstruct = sorted_ins(j);
            if (cstruct.score < top_score * ratio(Cls)), break; end
            final_per_class{Cls}(end+1) = cstruct;
            %% Count Information
            lowest_score(Cls) = min(lowest_score(Cls), scores(j));
            highst_score(Cls) = max(highst_score(Cls), scores(j));
            if (sum(image_roidb_train(cstruct.img_id).image_label == Cls) > 0)
                match_for_class(Cls) = match_for_class(Cls) + 1;
            end
        end
        sel_for_class(Cls) = numel(final_per_class{Cls});
    end
    clear score* temp ctemp idx sorted_ins Cls j jj top_score cstruct;;

    %% Print class with Thresh
    for Cls = 1:num_classes
        fprintf('[%02d] [%12s] : [FINAL= (OK) %3d/%3d] : [SEL= %3d/%3d] : Accuracy : %.4f :| score : [%.2f, %.2f]\n', Cls, conf.classes{Cls}, ...
                 match_for_class(Cls), sel_for_class(Cls), OPPO_SEL_PER_CLS(Cls), SELF_SEL_PER_CLS(Cls), ...
                 match_for_class(Cls) / sel_for_class(Cls), lowest_score(Cls), highst_score(Cls));
    end
    fprintf('Total Select : %-5d/%5d, Accuracy : %.4f\n', sum(match_for_class), sum(sel_for_class), sum(match_for_class) / sum(sel_for_class));
    clear match_for_class sel_for_class num Cls j OPPO_COUNT_PER_CLS SELF_SEL_PER_CLS lowest_score highst_score idx jj select_per_class;

    final_per_class = cat(2, final_per_class{:})';
    final_per_class = convert2image( final_per_class, num_roidb );
    assert (rem(num_roidb,2) == 0);
    %% Generate New
    new_image_roidb_train = [];
    for index = 1:num_roidb
        if (numel(final_per_class{index}) == 0), continue; end
        if (index <= num_roidb/2), reverse_idx = index + (num_roidb/2);
        else                       reverse_idx = index - (num_roidb/2); end
        if (index <= num_roidb/2)
            image_id = image_roidb_train(index).image_id;
            assert (strcmp([image_id, '_flip'], image_roidb_train(reverse_idx).image_id));
        end

        [pos_boxes, pos_cls, pos_scores, pos_regboxes] = abstra_struct(final_per_class{index}, index);
        if (conf.debug && index <= num_roidb/2) %%% Print Images with boxes
          weakly_debug(conf, image_roidb_train(index), pos_boxes, pos_regboxes, pos_cls, pos_scores, debug_dir);
        end

        rois        = image_roidb_train(index).boxes;
        num_boxes   = size(rois, 1);
        overlap     = zeros(num_boxes, num_classes, 'single');
        for bid = 1:numel(pos_cls)
            gt_classes = pos_cls(bid);
            gt_boxes   = pos_boxes(bid, :);
            overlap(:, gt_classes) = max(overlap(:, gt_classes), boxoverlap(rois, gt_boxes));
        end
        %append_bbox_regression_targets
        [bbox_targets, is_valid] = weakly_compute_targets(conf, rois, overlap);
        if(is_valid == false), continue; end
        new_image_roidb_train{end+1}            = image_roidb_train(index);
        new_image_roidb_train{end}.overlap      = overlap;
        new_image_roidb_train{end}.bbox_targets = bbox_targets;
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
    close all;
    fprintf('Generate new_image_roidb_train : %d, Cost : %.1f s\n', num_images, toc);
end

function ok = check_box(gt_box, boxes, thresh)
  overlap = boxoverlap(boxes, gt_box);
  %% Cover all ground truth
  gt_over = max(overlap);
  ok      = all(gt_over >= thresh);
  %% All boxes cover gt
  box_ov  = max(overlap, [], 2);
  ok      = ok && all(box_ov >= thresh);
end

function ok = check_cls(clsesA, clsesB)
  clsesA = unique(clsesA);
  clsesB = unique(clsesB);
  if (numel(clsesA) ~= numel(clsesB))
    ok = false;
  else
    ok = all(clsesA==clsesB);
  end
end

function [boxes, cls, scores, reg_boxes] = abstra_struct(structes, index)
  boxes = {structes.box};     boxes = cat(1, boxes{:});
  scores= {structes.score};   scores= cat(1, scores{:});
  cls   = {structes.cls};     cls   = cat(1, cls{:});
  reg_boxes = {structes.reg_box}; reg_boxes = cat(1, reg_boxes{:});
  img_ids = {structes.img_id}; img_ids = cat(1, img_ids{:});
  assert(all(img_ids==index));
end

function all_boxes = merge_boxes(all_boxes, x)
  if (isempty(all_boxes))
    all_boxes = x;
  else
    has = false;
    for j = 1:numel(all_boxes)
      assert(all_boxes(j).img_id == x.img_id);
      if (all_boxes(j).box_id == x.box_id)
        all_boxes(j).score = all_boxes(j).score + x.score;
        has = true;
      end
    end
    if (has == false)
      all_boxes(end+1) = x;
    end
  end
end
