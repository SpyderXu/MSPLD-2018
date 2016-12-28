function ok = filter_img_multibox(conf, caffe_test_net, image_path, boxes, cls, ori_score, thresh)
  max_rois_num_in_gpu = 10000;
  num_boxes = size(boxes, 1);
  [Tboxes, Tscores] = weakly_im_detect(conf, caffe_test_net, imread(image_path), multibox(boxes), max_rois_num_in_gpu);
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
