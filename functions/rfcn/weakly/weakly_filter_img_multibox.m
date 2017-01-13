function ok = weakly_filter_img_multibox(conf, caffe_test_net, image_path, structes);
  max_rois_num_in_gpu = 10000;
  boxes = {structes.box};     boxes = cat(1, boxes{:});
  scores= {structes.score};   scores= cat(1, scores{:});
  cls   = {structes.cls};     cls   = cat(1, cls{:});
  num_boxes = size(boxes, 1);
  test_boxes = multibox(boxes);
  [Tboxes, Tscores] = weakly_im_detect(conf, caffe_test_net, imread(image_path), test_boxes, max_rois_num_in_gpu);
  ok = true(num_boxes, 1);
  assert (rem(size(test_boxes,1), num_boxes) == 0);
  for idx = 1:size(test_boxes,1)/num_boxes
	cscores = Tscores((idx-1)*num_boxes+1 : idx*num_boxes, :);
	[mx_score, mx_cls] = max(cscores, [], 2);
    for j = 1:numel(cls)
	  if(mx_cls(j) == cls(j) && mx_score(j) > scores(j))
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

  CUR = boxes; W = (CUR(:,3)-CUR(:,1))/4; H = (CUR(:,4)-CUR(:,2))/4;
  CUR(:,1) = CUR(:,1) + W; CUR(:,3) = CUR(:,3) - W;
  CUR(:,2) = CUR(:,2) + H; CUR(:,4) = CUR(:,4) - H;
  ANS = [ANS;CUR];

  CUR = boxes; 
  CUR(:,1) = CUR(:,1) - W; CUR(:,3) = CUR(:,3) + W;
  CUR(:,2) = CUR(:,2) - H; CUR(:,4) = CUR(:,4) + H;
  ANS = [ANS;CUR];
  boxes = ANS;
end
