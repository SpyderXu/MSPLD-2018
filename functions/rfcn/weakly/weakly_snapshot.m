function weakly_snapshot(caffe_solver, bbox_means, bbox_stds, model_path)
%    file_name = [file_name, '.caffemodel'];
    bbox_pred_layer_name = 'rfcn_bbox';
    weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
    biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
    weights_back = weights;
    biase_back = biase;

    rep_time = size(weights, 4)/length(bbox_means(:));

    bbox_stds_flatten = bbox_stds';
    bbox_stds_flatten = bbox_stds_flatten(:);
    bbox_stds_flatten = repmat(bbox_stds_flatten, [1,rep_time])';
    bbox_stds_flatten = bbox_stds_flatten(:);
    bbox_stds_flatten = permute(bbox_stds_flatten, [4,3,2,1]);

    bbox_means_flatten = bbox_means';
    bbox_means_flatten = bbox_means_flatten(:);
    bbox_means_flatten = repmat(bbox_means_flatten, [1,rep_time])';
    bbox_means_flatten = bbox_means_flatten(:);
    bbox_means_flatten = permute(bbox_means_flatten, [4,3,2,1]);

    % merge bbox_means, bbox_stds into the model
    weights = bsxfun(@times, weights, bbox_stds_flatten); % weights = weights * stds; 
    biase = biase .* bbox_stds_flatten(:) + bbox_means_flatten(:); % bias = bias * stds + means;

    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

    %model_path = fullfile(cache_dir, file_name);
    caffe_solver.net.save(model_path);
    fprintf('Saved as %s\n', model_path);

    % restore net to original state
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
    caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end
