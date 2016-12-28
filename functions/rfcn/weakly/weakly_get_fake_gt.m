function new_image_roidb_train = weakly_get_fake_gt(conf, caffe_solver, image_roidb_train, bbox_means, bbox_stds, lambda, gamma, mx_select)
    if (mx_select <= 0), new_image_roidb_train = []; return; end
    mx_select = ceil(mx_select);
    assert (conf.flip); % argument
    caffe_test_net = caffe_solver.test_nets(1);
    caffe_test_net.set_phase('test');
    % determine the maximum number of rois in testing 

    num_roidb        = numel(image_roidb_train);       assert (rem(num_roidb,2) == 0);
    num_classes      = numel(conf.classes);
    select_per_class = cell(num_classes, num_roidb/2);
    tic;
    for index = 1:(num_roidb/2)
        if (rem(index, 500) == 0 || index == num_roidb/2)
            fprintf('Handle %3d / %3d image_roidb_train, cost : %.1f s\n', index, numel(image_roidb_train), toc);
        end
        reverse_idx = index + (num_roidb/2);

        %if (reverse_idx > num_roidb), reverse_idx = index - (num_roidb/2); end
        cur_roidb_train = [image_roidb_train(index), image_roidb_train(reverse_idx)];
        cur_img_ids     = [index, reverse_idx];
        [structs] = weakly_extract_struct(conf, caffe_test_net, cur_roidb_train, cur_img_ids, num_classes);
        assert (numel(structs) == num_classes);
        for cls = 1:num_classes
          %%select_per_class{structs(jj).cls}(end+1) = structs(jj);
          select_per_class(cls, index) = structs(cls);
        end
    end
    temp = cell(num_classes, 1);
    for cls = 1:num_classes
        ctemp = select_per_class(cls, :);
        ctemp = cat(1, ctemp{:});
        temp{cls} = ctemp;
    end
    select_per_class = temp;

    sel_for_class   = zeros(num_classes, 1);
    match_for_class = zeros(num_classes, 1);
    lowest_loss     = zeros(num_classes, 1) + inf;
    highst_loss     = zeros(num_classes, 1);

    for Cls = 1:num_classes
        if (numel(select_per_class{Cls}) == 0), continue; end
        temp   = select_per_class{Cls};
        losses = {temp.loss}; losses = cat(1, losses{:}); [~, idx] = sort(losses);
        sorted_images = temp(idx);
        select = [];
        for j = 1:numel(sorted_images)
            struct = sorted_images(j);
            assert (struct.cls == Cls);
            loss = struct.loss;
            if ( loss < lambda + gamma / (sqrt(j)+sqrt(j-1)) ) %%% Sampled
                %% Insert 
                %% For_Each_Img{struct.img_id}(end+1) = struct;
                select{end+1} = struct;
                %% Count Information
                sel_for_class(struct.cls) = sel_for_class(struct.cls) + 1;
                lowest_loss(struct.cls) = min(lowest_loss(struct.cls), loss);
                highst_loss(struct.cls) = max(highst_loss(struct.cls), loss);
                if (sum(image_roidb_train(struct.img_id).image_label == struct.cls) > 0)
                    match_for_class(struct.cls) = match_for_class(struct.cls) + 1;
                    %% Box match 
                    %image_index = sorted_images(j).index;
                    %boxes_index = sorted_images(j).box_idx;
                    %gt_boxes    = image_roidb_train(image_index).boxes(boxes_index, :);
                    %if ( check_box(sorted_images(j).Debug.gt_box, gt_boxes, 0.2) )
                    %    true_box_match(sorted_images(j).mx_class) = true_box_match(sorted_images(j).mx_class) + 1;
                    %end
                end
            end
        end
        select_per_class{Cls} = select;
    end
    clear loss* temp ctemp idx sorted_images Cls j jj;

    %MX_Select = inf;
    %for index = 1:num_classes
    %    MX_Select = min(MX_Select, numel(select_per_class{index}));
    %end
    MX_Select = mx_select;
    %MX_Select = inf;
    %MX_Select = max(MX_Select + 10, 2 * MX_Select);
    %MX_Select = min(MX_Select + conf.upper_for_min, 200);
    %MX_Select = min(MX_Select + conf.upper_for_min, 600);
    Fss_for_class   = zeros(num_classes, 1);
    Ftt_for_class   = zeros(num_classes, 1);
    
    For_Each_Img    = cell(num_roidb, 1);
    thresh_per_cls  = zeros(num_classes, 1) + inf;
    for Cls = 1:num_classes
        Fss_for_class(Cls) = min(MX_Select, numel(select_per_class{Cls}));
        %picked = randperm(min(numel(select_per_class{Cls}),2*Fss_for_class(Cls)), Fss_for_class(Cls));
        picked = randperm(numel(select_per_class{Cls}), Fss_for_class(Cls));
        unpick = setdiff((1:numel(select_per_class{Cls})), picked);
        for j = 1:numel(picked)
            struct         = select_per_class{Cls}{picked(j)};
            struct.pick    = true;
            For_Each_Img{struct.img_id}(end+1) = struct;
            thresh_per_cls(struct.cls) = min(thresh_per_cls(struct.cls), struct.score);
            assert (Cls == struct.cls);
            if (sum(image_roidb_train(struct.img_id).image_label == struct.cls) > 0)
                Ftt_for_class(struct.cls) = Ftt_for_class(struct.cls) + 1;
            end
        end
        for j = 1:numel(unpick)
            struct         = select_per_class{Cls}{unpick(j)};
            struct.pick    = false;
            For_Each_Img{struct.img_id}(end+1) = struct;
            thresh_per_cls(struct.cls) = min(thresh_per_cls(struct.cls), struct.score);
        end
    end

    %% Print class with Thresh
    for Cls = 1:num_classes
        fprintf('SELECT [%02d] [%12s] : D[OK : %4d / Sel: %4d] : [OK : %4d / Sel: %4d] :| loss : [%.2f, %.2f],  score : [%.2f, %.2f]\n', Cls, conf.classes{Cls}, ...
		 Ftt_for_class(Cls), Fss_for_class(Cls), match_for_class(Cls), sel_for_class(Cls), ...
                 lowest_loss(Cls), highst_loss(Cls), exp(-lowest_loss(Cls)), exp(-highst_loss(Cls)));
    end
    clear MX_Select Fss_for_class Ftt_for_class match_for_class sel_for_class num Cls j;

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
        %%% Filter Picked
        Final_Struct = [];
        ok = false;
        for jj = 1:numel(For_Each_Img{index})
          struct = For_Each_Img{index}(jj);
          if (struct.score >= thresh_per_cls(struct.cls) || struct.pick == true)
            Final_Struct{end+1} = struct;
          end
          if (struct.pick == true)
            ok = true;
          end
        end
        if (isempty(Final_Struct) || ok == false), continue;end
        For_Each_Img{index} = cat(1, Final_Struct{:});

        [pos_boxes, pos_cls, pos_scores, pos_regboxes] = abstra_struct(For_Each_Img{index});
        %[rev_boxes, rev_cls] = abstra_struct(For_Each_Img{reverse_idx});
        %if (check_cls(pos_cls, rev_cls)), continue; end
        if (conf.debug) %%% Print Images with boxes
          weakly_debug(conf, image_roidb_train(index), pos_boxes, pos_regboxes, pos_cls, pos_scores, num2str(caffe_solver.iter()));
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
        for j = 1:numel(pos_cls)
            Sel_After_argument(pos_cls(j)) = Sel_After_argument(pos_cls(j)) + 1;
            if (sum(image_roidb_train(index).image_label == pos_cls(j)) > 0)
                Tru_After_argument(pos_cls(j)) = Tru_After_argument(pos_cls(j)) + 1;
            end
        end
    end
    for Cls = 1:num_classes
        fprintf('FINAL  [%02d] [%12s] : [OK : %4d / Sel: %4d] \n', Cls, conf.classes{Cls},...
		 Tru_After_argument(Cls), Sel_After_argument(Cls));
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
    fprintf('Generate new_image_roidb_train : %d, lambda : %.2f, gamma : %.2f, Cost : %.1f s\n', num_images...
                    ,lambda, gamma,toc);
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
   boxes     = zeros(numel(structs), 4); 
   reg_boxes = zeros(numel(structs), 4); 
   cls       = zeros(numel(structs), 1);
   scores    = zeros(numel(structs), 1);
   for idx = 1:numel(structs)
       boxes(idx, :)     = structs(idx).box;
       reg_boxes(idx, :) = structs(idx).reg_box;
       cls(idx)          = structs(idx).cls;
       scores(idx)       = structs(idx).score;
   end
end
