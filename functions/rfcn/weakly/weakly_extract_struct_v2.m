function [structs] = weakly_extract_struct_v2(conf, caffe_test_net, image_roidb_train, img_ids, num_classes, thresh_hold)
  max_rois_num_in_gpu = 10000;
  assert (numel(image_roidb_train) == 1 || numel(image_roidb_train) == 2);
  assert (numel(image_roidb_train) == numel(img_ids));
  box_num = size(image_roidb_train(1).boxes, 1);
  if (numel(image_roidb_train) == 2)
    assert (all(image_roidb_train(1).im_size == image_roidb_train(2).im_size));
    reverse_box = image_roidb_train(1).boxes;
    reverse_box(:, [1,3]) = image_roidb_train(1).im_size(2) + 1 - reverse_box(:, [3,1]);
    assert (all(all(reverse_box == image_roidb_train(2).boxes)));
  end

  [boxes, scores]        = weakly_im_detect(conf, caffe_test_net, imread(image_roidb_train(1).image_path), image_roidb_train(1).boxes, max_rois_num_in_gpu);

  if (numel(image_roidb_train) == 2)
    [rev_boxes, rev_scores]        = weakly_im_detect(conf, caffe_test_net, imread(image_roidb_train(2).image_path), image_roidb_train(1).boxes, max_rois_num_in_gpu);
    rev_boxes(:, [1,3]) = image_roidb_train(1).im_size(2) + 1 - rev_boxes(:, [3,1]);
    boxes  = (boxes + rev_boxes) / 2;
    scores = (scores+ rev_scores) / 2;
  end
  [MX_per_class, ID_cls] = max(scores);
  [MX_per_boxes, ID_bbx] = max(scores, [], 2);
  [mx_score, mx_class]   = max(MX_per_class);
  assert (num_classes == size(scores, 2));
  %thresh_hold            = 0.01;
  structs = cell(num_classes, 1);
  for Cls = 1:num_classes
    %if (ID_bbx(ID_cls(Cls)) ~= Cls), continue; end
    aboxes         = [image_roidb_train(1).boxes, scores(:,Cls)];
    keep           = nms(aboxes, 0.3);
    keep           = keep(ID_bbx(keep)==Cls);
    keep           = keep(scores(keep,Cls) >= thresh_hold);
    cstructs       = [];
    for j = 1:numel(keep)
        box_id = keep(j);
        box_score      = aboxes(box_id, end);
        struct         = [];
        struct.score   = box_score;
        struct.cls     = Cls;
        struct.box     = image_roidb_train(1).boxes(box_id, :);
        struct.reg_box = boxes(box_id, (Cls-1)*4+1:Cls*4);
        struct.img_id  = img_ids(1);
        struct.box_id  = box_id;
        cstructs{end+1} = struct;
        if (numel(image_roidb_train) == 2)
          struct.img_id= img_ids(2);
          struct.box   = image_roidb_train(2).boxes(box_id, :);
          struct.reg_box(:, [1,3]) = image_roidb_train(1).im_size(2) + 1 - struct.reg_box(:, [3,1]);
          cstructs{end+1} = struct;
        end
    end
    if(isempty(cstructs) == false), cstructs = cat(1, cstructs{:}); end
    structs{Cls} = cstructs;
  end
  structs = cat(1, structs{:});
end
