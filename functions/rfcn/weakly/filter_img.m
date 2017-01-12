function ANS = filter_img(All_Img_Struct)
  ANS = {};
  for idx = 1:numel(All_Img_Struct)
     x = All_Img_Struct(idx);
     cls = x.cls;
	 %if (numel(cls) > 7 || numel(cls) > numel(unique(cls)))
	 if (numel(cls) > numel(unique(cls)))
		continue;
	 end
	 box = x.box;
	 if (any((box(:,3)-box(:,1)) < 25) || any((box(:,4)-box(:,2)) < 25))
        continue;
     end
     ANS{end+1} = x;
  end
  ANS = cat(1, ANS{:});
end
