function All_Img_Struct = filter_struct(All_Struct, thresh)
  All_Img_Struct = [];
  for index = 1:numel(All_Struct)
    img_structs = weakly_convert_structs2img(All_Struct{index}, thresh);
    All_Img_Struct = [All_Img_Struct;img_structs];
  end
  All_Img_Struct = cat(1, All_Img_Struct{:});
end
