function ANS = filter_img(All_Img_Struct)
  ANS = {};
  for idx = 1:numel(All_Img_Struct)
     x = All_Img_Struct(idx);
     cls = x.cls;
	 if (numel(cls) > 5 || numel(cls) > numel(unique(cls)))
		continue;
	 end
	 box = x.box;
	 if (any((box(:,3)-box(:,1)) < 30) || any((box(:,4)-box(:,2)) < 30))
        continue;
     end
     ANS{end+1} = x;
  end
  ANS = cat(1, ANS{:});
end
