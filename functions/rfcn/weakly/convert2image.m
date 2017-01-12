function all_image = convert2image( box_selected, num_roidb)
  all_image = cell(num_roidb, 1);
  for idx = 1:numel(box_selected)
    img_ids = box_selected(idx).img_id;
    assert (img_ids <= num_roidb);
    all_image{img_ids}(end+1) = box_selected(idx);
  end
end
