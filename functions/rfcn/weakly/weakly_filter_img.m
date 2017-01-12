function Filterred = weakly_filter_img(All_Img_Struct, classes, smallest)
  Filterred = cell(numel(All_Img_Struct), 1);
  PerClsNum = zeros(numel(classes), 1);
  count = 0;
  for idx = 1:numel(All_Img_Struct)
     all_boxes = All_Img_Struct{idx};
     if (isempty(all_boxes)), continue; end
     cls = {all_boxes.cls}; cls = cat(1, cls{:});
     if (numel(cls) > numel(unique(cls))), continue; end
     assert (all(cls<=numel(classes)));
     boxes = {all_boxes.reg_box}; boxes = cat(1, boxes{:});
     keepB = find(boxes(:,3)-boxes(:,1) >= smallest);
     keepA = find(boxes(:,4)-boxes(:,2) >= smallest);
     keep  = intersect(keepA, keepB);
     if (isempty(keep)), continue; end
     Filterred{idx} = all_boxes(keep);
     count = count + 1;
     PerClsNum(cls(keep)) = PerClsNum(cls(keep)) + 1;
  end
  fprintf('weakly_filter_img : %4d / %4d\n', count, numel(All_Img_Struct));
  for idx = 1:numel(classes)
    fprintf('---=== %12s has %d pictures\n', classes{idx}, PerClsNum(idx));
  end
end
