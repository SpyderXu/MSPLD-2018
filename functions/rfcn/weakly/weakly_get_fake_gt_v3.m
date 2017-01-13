function new_image_roidb_train = weakly_get_fake_gt_v3(conf, caffe_test_net, image_roidb_train, bbox_means, bbox_stds, PER_Select, debug_dir)
    assert (conf.flip); % argument

    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(conf.classes);   
    tic;
    All_Select = cell(num_roidb/2, 1);
    for index = 1:(num_roidb/2)
        if (rem(index, 500) == 0 || index == num_roidb/2)
            fprintf('Handle %3d / %3d image_roidb_train, cost : %.1f s\n', index, num_roidb/2, toc);
        end
        reverse_idx = index + (num_roidb/2);
        %if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        cur_roidb_train = [image_roidb_train(index), image_roidb_train(reverse_idx)];
        cur_img_ids     = [index, reverse_idx];
        [structs] = weakly_extract_struct(conf, caffe_test_net, cur_roidb_train, cur_img_ids, num_classes);
        assert (numel(structs) == num_classes);
        All_Select{index} = structs;
    end
    All_Img_Struct = filter_struct(All_Select, 0.2);
    All_Img_Struct = filter_img(All_Img_Struct);
    sampled_ids = [];
    All_OKs = {};
    tic;
    for idx = 1:numel(All_Img_Struct)
		struct = All_Img_Struct(idx);
		image_path = image_roidb_train(struct.img_id).image_path;
		ok = filter_img_multibox(conf, caffe_test_net, image_path, struct.box, struct.cls, struct.score, 0.0);
		if (all(ok)), sampled_ids(end+1) = idx; end
		if (rem(idx, 500) == 0 || idx == numel(All_Img_Struct)), fprintf('Handle %4d / %4d, cost %.1f s \n', idx, numel(All_Img_Struct), toc); end
    end
    All_Img_Struct = All_Img_Struct(sampled_ids);

    select_per_class = cell(num_classes, 1);
    INIT_COUNT_PER_CLS = zeros(num_classes, 1);
	for idx = 1:numel(All_Img_Struct)
      clss = All_Img_Struct(idx).cls;
	  score = All_Img_Struct(idx).score;
	  assert (numel(clss) == numel(score));
      for ii = 1:numel(clss)
		struct = [];
		struct.idx = idx;
		struct.score = score(ii);
		cls = clss(ii);
		select_per_class{cls} = [select_per_class{cls}; struct];
      end
    end
	for cls = 1:num_classes
	  INIT_COUNT_PER_CLS(cls) = numel(select_per_class{cls});
    end

    MIN_Select      = min(PER_Select);
    MAX_SEL_PER_CLS = (INIT_COUNT_PER_CLS + MIN_Select);
    MAX_SEL_PER_CLS = (MAX_SEL_PER_CLS / min(MAX_SEL_PER_CLS)) * MIN_Select;
    MAX_SEL_PER_CLS = min(ceil(MAX_SEL_PER_CLS), INIT_COUNT_PER_CLS);
    ORIGINAL_SEL_MX = MAX_SEL_PER_CLS;
    PER_Select      = ceil(PER_Select);
    MAX_SEL_PER_CLS = max([MAX_SEL_PER_CLS, PER_Select], [], 2);
    LIMIT = 10 * min(MAX_SEL_PER_CLS);
    MAX_SEL_PER_CLS(find(MAX_SEL_PER_CLS>=LIMIT)) = LIMIT;

    sel_for_class   = zeros(num_classes, 1);
    match_for_class = zeros(num_classes, 1);
    lowest_score    = zeros(num_classes, 1) + inf;
    highst_score    = zeros(num_classes, 1);

    final_per_class = cell(num_classes, 1);
    %ratio = [0.7, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.7, 0.0, 0.5, ...
    %         0.0, 0.7, 0.7, 0.0, 0.7, 0.0, 0.0, 0.7, 0.7, 0.0];
    ratio = zeros(num_classes, 1);
    for Cls = 1:num_classes
        if (numel(select_per_class{Cls}) == 0), continue; end
        temp   = select_per_class{Cls};
        scores = {temp.score}; scores = cat(1, scores{:}); [scores, idx] = sort(scores, 'descend');
        sorted_ins = temp(idx);
        select = [];
        top_score = scores(1);
        for j = 1:min(MAX_SEL_PER_CLS(Cls), numel(sorted_ins))
            struct = sorted_ins(j);
            if (struct.score < top_score * ratio(Cls)), break; end
            select{end+1} = struct;
            %% Count Information
            lowest_score(Cls) = min(lowest_score(Cls), scores(j));
            highst_score(Cls) = max(highst_score(Cls), scores(j));
            if (sum(image_roidb_train(All_Img_Struct(struct.idx).img_id).image_label == Cls) > 0)
                match_for_class(Cls) = match_for_class(Cls) + 1;
            end
        end
		sel_for_class(Cls) = numel(select);
        final_per_class{Cls} = select;
    end
    clear score* temp ctemp idx sorted_ins Cls j jj;

    For_Each_Img  = cell(num_roidb, 1);
    for Cls = 1:num_classes
		  for j = 1:numel(final_per_class{Cls})
			  struct = final_per_class{Cls}{j};
			  struct = All_Img_Struct(struct.idx);
			  For_Each_Img{struct.img_id} =  merge_struct(For_Each_Img{struct.img_id}, struct, Cls);
		  end
	  end
    FINAL_PER_CLS = zeros(num_classes, 1);
    for idx = 1:numel(For_Each_Img)
      if (isempty(For_Each_Img{idx})), continue; end
	    for jj = 1:numel(For_Each_Img{idx}.cls)
		    cls = For_Each_Img{idx}.cls(jj);
        FINAL_PER_CLS(cls) = FINAL_PER_CLS(cls) + 1;
	    end
    end

    %% Print class with Thresh
    for Cls = 1:num_classes
        fprintf('[%02d] [%12s] : FINAL : %3d = [INIT : %3d, MX_SEL : (%3d,%3d)=%3d] : [OK : %4d / Sel: %4d] Accuracy : %.4f :| score : [%.2f, %.2f]\n', Cls, conf.classes{Cls}, ...
		    FINAL_PER_CLS(Cls), INIT_COUNT_PER_CLS(Cls), PER_Select(Cls), ORIGINAL_SEL_MX(Cls), MAX_SEL_PER_CLS(Cls), ...
            match_for_class(Cls), sel_for_class(Cls), match_for_class(Cls) / sel_for_class(Cls), lowest_score(Cls), highst_score(Cls));
    end
	  fprintf('Total Select : %5d , OK : %5d , Accuracy : %.4f\n', sum(sel_for_class), sum(match_for_class), sum(match_for_class) / sum(sel_for_class));
    clear match_for_class sel_for_class num Cls j FINAL_PER_CLS INIT_COUNT_PER_CLS MAX_SEL_PER_CLS lowest_score highst_score idx jj final_per_class select_per_class ORIGINAL_SEL_MX;

    assert (rem(num_roidb,2) == 0);
    %% Generate New
    Sel_After_argument  = zeros(num_classes, 1);
    Tru_After_argument  = zeros(num_classes, 1);
    new_image_roidb_train = [];
    for index = 1:num_roidb
        if (numel(For_Each_Img{index}) == 0), continue; end
        reverse_idx = index + (num_roidb/2);
        if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        if (index < num_roidb/2)
            image_id = image_roidb_train(index).image_id;
            assert (strcmp([image_id, '_flip'], image_roidb_train(reverse_idx).image_id));
        end

        [pos_boxes, pos_cls, pos_scores, pos_regboxes] = abstra_struct(For_Each_Img{index});
        %[rev_boxes, rev_cls] = abstra_struct(For_Each_Img{reverse_idx});
        %if (check_cls(pos_cls, rev_cls)), continue; end
        if (conf.debug && index < num_roidb/2) %%% Print Images with boxes
          weakly_debug(conf, image_roidb_train(index), pos_boxes, pos_regboxes, pos_cls, pos_scores, debug_dir);
        end

        num_boxes   = size(image_roidb_train(index).boxes, 1);
        overlap     = zeros(num_boxes, num_classes, 'single');
        for bid = 1:numel(pos_cls)
            gt_classes = pos_cls(bid);
            gt_boxes   = pos_boxes(bid, :);
            overlap(:, gt_classes) = max(overlap(:, gt_classes), boxoverlap(image_roidb_train(index).boxes, gt_boxes));
        end
        %append_bbox_regression_targets
        rois        = image_roidb_train(index).boxes;
        [bbox_targets, is_valid] = weakly_compute_targets(conf, rois, overlap);
        if is_valid == false
            continue;
        end
        new_image_roidb_train{end+1}            = image_roidb_train(index);
        new_image_roidb_train{end}.overlap      = overlap;
        new_image_roidb_train{end}.bbox_targets = bbox_targets;
        %% Count Information
        %for j = 1:numel(pos_cls)
        %    Sel_After_argument(pos_cls(j)) = Sel_After_argument(pos_cls(j)) + 1;
        %    if (sum(image_roidb_train(index).image_label == pos_cls(j)) > 0)
        %        Tru_After_argument(pos_cls(j)) = Tru_After_argument(pos_cls(j)) + 1;
        %    end
        %end
    end
    %for Cls = 1:num_classes
    %    fprintf('FINAL  [%02d] [%12s] : [OK : %4d / Sel: %4d] \n', Cls, conf.classes{Cls}, Tru_After_argument(Cls), Sel_After_argument(Cls));
    %end

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

function [boxes, cls, scores, reg_boxes] = abstra_struct(structs)
   boxes     = structs.box;
   reg_boxes = structs.reg_box;
   cls       = structs.cls;
   scores    = structs.score;
end

function ANS = merge_struct(old, struct, cls)
   if (isempty(old) == 0)
      cls = [cls;old.cls];
   end
   idx = [];
   for index = 1:numel(cls)
	 ccls = cls(index);
	 idx = [idx; find(struct.cls==ccls)];
   end
   struct.loss = struct.loss(idx);
   struct.score = struct.score(idx);
   struct.cls = struct.cls(idx);
   struct.box = struct.box(idx, :);
   struct.reg_box = struct.reg_box(idx, :);
   ANS = struct;
end
