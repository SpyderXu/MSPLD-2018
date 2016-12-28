function weakly_debug(conf, image_roidb, boxes, reg_boxes, cls, scores, dir_name)
  if (isfield(conf, 'debug') == false || conf.debug == false), return; end
  assert (size(boxes,1) == size(cls,1) && size(cls,1) == size(scores,1));
  classes = conf.classes;
  boxes_cell = cell(length(classes), 1);
  for i = 1:length(classes)
    boxes_cell{i} = zeros(0,5);
  end
  reg_boxes_cell = boxes_cell;
  grt_boxes_cell = boxes_cell;
  for i = 1:length(cls)
    boxes_cell{cls(i)}     = [boxes_cell{cls(i)};     [boxes(i,:), scores(i)]];
    reg_boxes_cell{cls(i)} = [reg_boxes_cell{cls(i)}; [reg_boxes(i,:), scores(i)]];
  end
  mkdir_if_missing(fullfile(conf.debug_cache_dir, dir_name));
  %figure(1);
  im = imread(image_roidb.image_path);
  showboxes(im, boxes_cell, classes, 'voc');
  split_name = image_roidb.image_id;
  saveas(gcf, fullfile(conf.debug_cache_dir, dir_name, [split_name, '.jpg']));
%%% Regression Boxes
  %showboxes(im, reg_boxes_cell, classes, 'voc');
  %saveas(gcf, fullfile(conf.debug_cache_dir, dir_name, [split_name, '_REG.jpg']));
  
%%% Draw Ground Truth
  for i = 1:length(image_roidb.Debug_GT_Cls)
    grt_boxes_cell{image_roidb.Debug_GT_Cls(i)} = [grt_boxes_cell{image_roidb.Debug_GT_Cls(i)}; [image_roidb.Debug_GT_Box(i,:), 1]];
  end
  %figure(1);
  showboxes(im, grt_boxes_cell, classes, 'voc');
  saveas(gcf, fullfile(conf.debug_cache_dir, dir_name, [split_name, '_GRT.jpg']));
  
end
