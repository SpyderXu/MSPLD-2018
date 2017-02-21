function mode = weakly_train_mode( conf )
    if ( conf.regression ) 
        mode = 0; %% supervise rfcn training 
    else
        mode = 1; %% supervise score only train
    end
end
