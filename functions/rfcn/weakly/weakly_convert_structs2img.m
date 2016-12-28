function img_struct = weakly_convert_structs2img(cstructs, thresh)
  img_struct = {};
  img_ids = [];
  for index = 1:numel(cstructs)
    xx = cstructs{index};
	for j = 1:numel(xx)
		x = xx(j);
		img_id = x.img_id;
		if (x.score < thresh), continue;end
		id = find(img_ids == img_id);
		if (isempty(id))
			id = numel(img_ids)+1;
			struct = x;
		elseif (numel(id) == 1)
			struct = img_struct{id};
			struct.loss = [struct.loss; x.loss];
			struct.score = [struct.score; x.score];
			struct.box = [struct.box; x.box];
			struct.cls = [struct.cls; x.cls];
			struct.reg_box = [struct.reg_box; x.reg_box];
		else
			error('wrong');
		end
		img_ids(id) = img_id;
		img_struct{id} = struct;
	end
  end
  img_struct = img_struct';
end
