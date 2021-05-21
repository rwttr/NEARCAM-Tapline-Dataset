# Build nearcam DataFold
# import JSON # groundtruth file
# import CSV  # k-fold 
# import FileIO
# import Images
# import Statistics
# import Flux
# import ImageView # Visualize Sample

# global scope
# -- path to dataset
dataset_path = "E:/Thesis Dataset/NEARCAM Tapping Line Dataset" # default path
# -- shared Variable dispatchRecorder (state variable)
dispatch_indx_record = 1;
# Dataset image size
const annotated_img_width = 1280;
const annotated_img_height = 720;

# DataStore Struct contain images and groundtruth urls
struct DataStore
    data_url_rgb1::Vector{Any}       # rgb1 image url array
    data_url_rgb2::Vector{Any}       # rgb2 image url array
    data_url_d8::Vector{Any}         # depth-u8 image url array
    data_url_d16::Vector{Any}        # depth-u16 image url array
    label_bbox::Vector{Any}          # annotated bbox  
    label_px_imgURL::Vector{Any}     # annotated pixel image url
    n::Integer                       # row count on data               
end
# Extern DataFold structure contain training and testing data of a fold
struct DataFold
    training_data::DataStore
    test_data::DataStore
    fold_id::Integer                 # Fold ID
end

function buildDatafold(fold_no::Integer, dataset_path::String)
    # -------------- dataset directory structure ----------------
    # /images
    img_rgb1_dir    = joinpath(dataset_path, "images/RGB1");
    img_rgb2_dir    = joinpath(dataset_path, "images/RGB2");
    img_d8_dir      = joinpath(dataset_path, "images/Depth-u8");
    img_d16_dir     = joinpath(dataset_path, "images/Depth-u16");
    # /annotations
    taplineBbox = joinpath(dataset_path, "annotations/TaplineBox.json");
    taplinePixels_dir = joinpath(dataset_path, "annotations/TaplinePixels");
    # /kfold    K-fold separator (k=5), single fold selection specify in fold_no
    fold_train  = joinpath(dataset_path, "kfold/fold_$fold_no", "train.csv");
    # fold_test   = joinpath(dataset_path, "kfold/fold_$fold_no", "test.csv"); # already encoded in train.csv
    # ------------------------------------------------------
    # -- bbox GroundTruth
    gt_bbox = JSON.parsefile(taplineBbox);
    # note:
    # image_no :xxxx,    as the key of outter dict,
    # hb, wb, xb, yb,    as key for bbox value stored in inner dict
    # example
    #       gt_data["image_no"][".."]
    #       gt_data["0001"]["wb"]

    # Extract Dataset File Path 
    # kfold index arranged in single column (.Column1 property)
    train_indx = CSV.File(fold_train, header=false).Column1;
    # test_indx = CSV.File(fold_test, header=false).Column1;

    # Training Data
    # --Images
    training_img_rgb1 = [];
    training_img_rgb2 = [];
    training_img_d8 = [];
    training_img_d16 = [];
    # --Labels
    training_px_label = [];
    training_bbox_label = [];

    # Test Data
    # --Images
    test_img_rgb1 = [];
    test_img_rgb2 = [];
    test_img_d8 = [];
    test_img_d16 = [];
    # --Labels
    test_px_label = [];
    test_bbox_label = [];

    for i = 1:length(train_indx)
        img_id = lpad(string(i), 4, '0'); # apply leading zeros in image_name  
        img_name = img_id * ".png";  
        xb = gt_bbox[img_id]["xb"];
        yb = gt_bbox[img_id]["yb"];
        wb = gt_bbox[img_id]["wb"];
        hb = gt_bbox[img_id]["hb"];
    
        if train_indx[i] == 1        
        # training image path
            push!(training_img_rgb1, joinpath(img_rgb1_dir, img_name));
            push!(training_img_rgb2, joinpath(img_rgb2_dir, img_name));
            push!(training_img_d8, joinpath(img_d8_dir, img_name));
            push!(training_img_d16, joinpath(img_d16_dir, img_name));
        # tapline label image path
            push!(training_px_label, joinpath(taplinePixels_dir, img_name));
        # bbox in [x y w h]  row format
            push!(training_bbox_label, [xb yb wb hb;]);
        else
            push!(test_img_rgb1, joinpath(img_rgb1_dir, img_name));
            push!(test_img_rgb2, joinpath(img_rgb2_dir, img_name));
            push!(test_img_d8, joinpath(img_d8_dir, img_name));
            push!(test_img_d16, joinpath(img_d16_dir, img_name));
        # tapline label image path 
            push!(test_px_label, joinpath(taplinePixels_dir, img_name));
        # bbox in [x y w h]  row format
            push!(test_bbox_label, [xb yb wb hb;]);
        end
    end
    
    traningDataStore = DataStore(
        training_img_rgb1,
        training_img_rgb2,
        training_img_d8,
        training_img_d16,
        training_bbox_label,
        training_px_label,
        length(training_img_rgb1)
    );

    testDataStore = DataStore(
        test_img_rgb1,
        test_img_rgb2,
        test_img_d8,
        test_img_d16,
        test_bbox_label,
        test_px_label,
        length(test_img_rgb1)
    );

    dataFold = DataFold(
        traningDataStore,
        testDataStore,
        fold_no
    );

    return dataFold;
end

# build DataFold as Global Variable
function init(;datapath::String=dataset_path, kfold_enable::Bool=true) 
    global dataset_path = datapath;
    # Extern Dataset Arranged in Fold ::DataFold
    if kfold_enable
        # build k=5
        global dataFold_1 = buildDatafold(1, dataset_path);
        global dataFold_2 = buildDatafold(2, dataset_path);
        global dataFold_3 = buildDatafold(3, dataset_path);
        global dataFold_4 = buildDatafold(4, dataset_path);
        global dataFold_5 = buildDatafold(5, dataset_path);
    else
        # build single fold ; fold_id = 1
        global dataFold_single = buildDatafold(1, dataset_path);         
    end
end

function rescaleBBox(bbox, target_size::Matrix{Int64}, source_size::Matrix{Int64})
    # size in W , H
    w_factor = target_size[1] / source_size[1];
    h_factor = target_size[2] / source_size[2];    
    return bbox .* [w_factor h_factor w_factor h_factor]; # x y w h
end

function rescalePxImg(tapline_px_img, target_size::Matrix{Int64})
    # tapline_px_img : original size annotated image 
    # target_size = [W H] ; = column, row
    w = target_size[1];
    h = target_size[2];
    output_tapline_img = Bool.(zeros(h, w));
    rescale_factor_w = target_size[1] / annotated_img_width;
    rescale_factor_h = target_size[2] / annotated_img_height;

    tapline_px = findall(Bool.(tapline_px_img)); # source_px::CartesianIndex
    tapline_px_rescale = []; # rescaled_px::CartesianIndex

    for i = 1:length(tapline_px)
        temp_px_point = tapline_px[i].I;
        temp_px_point_rescaled = temp_px_point .* (rescale_factor_h, rescale_factor_w);
        temp_px_point_rescaled = round.(Int, temp_px_point_rescaled);

        # verify bound 
        if (temp_px_point_rescaled[1] <= h) && (temp_px_point_rescaled[2] <= w)
            push!(tapline_px_rescale,
                CartesianIndex(Int16(temp_px_point_rescaled[1]), Int16(temp_px_point_rescaled[2]))  
            );
        end
    end
    # eliminate redundant pixels
    tapline_px_rescale = unique(tapline_px_rescale);
    
    # draw output on image
    # for i = 1:length(tapline_px_rescale)
    #     output_tapline_img[tapline_px_rescale[i]] = 1;
    # end

    # Thining edges
    row_value = map(i -> i[1], tapline_px_rescale);
    col_value = map(i -> i[2], tapline_px_rescale);
    col_value_unq = unique(col_value);
    
    thin_tapline_px_rescale = [];

    for i = 1:length(col_value_unq)
        # count number of occurrence of col_value_unq in col_value
        # occur_count = count(x -> x == col_value_unq[i], col_value);
        occur_indx  = findall(x -> x == col_value_unq[i], col_value);

        # filter_row_value = round(sum(row_value[occur_indx]) / occur_count);
        filter_row_value = round(Statistics.mean(row_value[occur_indx]));

        # verify bound
        if filter_row_value <= h
            push!(thin_tapline_px_rescale, CartesianIndex(Int32.(filter_row_value), col_value_unq[i]));
        end
        # draw output image
        output_tapline_img[thin_tapline_px_rescale[i]] = 1;
    end

    return output_tapline_img;

end

function dispatchData(datafold::DataStore; 
    shuffle_enable::Bool=false,
    dispatch_size::Integer=1, 
    data_selector::String="rgb1", 
    img_outputsize=[128 128])

    # output Flux DataLoader with size of WHCB, batchsize = dispatch_size
    # signature: DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    w, h = img_outputsize;
    b = dispatch_size;  

    if data_selector == "rgb1"
        data_url = datafold.data_url_rgb1;
    elseif data_selector == "rgb2"
        data_url = datafold.data_url_rgb2;
    elseif data_selector == "d8"
        data_url = datafold.data_url_d8;
    else
        data_url = datafold.data_url_d16;
    end

    # dispatch image data : loaded_img  
    temp_img = Images.load(data_url[dispatch_indx_record]);
    temp_img_pp = Images.imresize(temp_img, w, h);
    temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
    loaded_img = copy(temp_img_pp);  
    loaded_img = PermutedDimsArray(loaded_img, (2, 3, 1)); # W.H.C
    loaded_img = Flux.unsqueeze(loaded_img, 4); # W.H.C.N

    # dispatch bbox : loaded_bbox
    loaded_bbox = copy(datafold.label_bbox[dispatch_indx_record]);
    loaded_bbox = rescaleBBox(loaded_bbox, [w h], [annotated_img_width annotated_img_height]);
    loaded_bbox = reshape(loaded_bbox, (1, 4, 1, 1)); # WHCN bbox 1 sample in [x y w h] format
 
    # dispatch px : loaded_px
    temp_img = Images.load(datafold.label_px_imgURL[dispatch_indx_record]);
    temp_img_pp = rescalePxImg(temp_img, [w h]);
    temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
    loaded_px = copy(temp_img_pp);
    loaded_px = reshape(loaded_px, (1, w, h)); # Channel x W x H 
    loaded_px = PermutedDimsArray(loaded_px, (2, 3, 1)); # WHC
    loaded_px = Flux.unsqueeze(loaded_px, 4); # W.H.C.N

    global dispatch_indx_record += 1;
    
    if (dispatch_size > 1) && (dispatch_indx_record <= datafold.n) # check data remaining
        for i = 2:dispatch_size

        # image data
            temp_img = Images.load(data_url[dispatch_indx_record]);
            temp_img_pp = Images.imresize(temp_img, w, h);
            temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
            temp_img_pp = PermutedDimsArray(temp_img_pp, (2, 3, 1)); # W.H.C
            temp_img_pp = Flux.unsqueeze(temp_img_pp, 4); # W.H.C.N
            loaded_img = cat(temp_img_pp, loaded_img; dims=4); # concatenate loaded image along dims=4 dimension
        
        # bbox
            temp_bbox = copy(datafold.label_bbox[dispatch_indx_record]);
            temp_bbox = rescaleBBox(temp_bbox, [w h], [annotated_img_width annotated_img_height]);
            temp_bbox = reshape(temp_bbox, (1, 4, 1, 1)); # WHCN bbox 1 sample in [x y w h] format
            loaded_bbox = cat(temp_bbox, loaded_bbox; dims=4);
        
        # px
            temp_img = Images.load(datafold.label_px_imgURL[dispatch_indx_record]);
            temp_img_pp = rescalePxImg(temp_img, [w h]);
            temp_img_pp = Images.channelview(temp_img_pp); # Channel x W x H
            temp_img_pp = reshape(loaded_px, (1, w, h));
            temp_img_pp = PermutedDimsArray(temp_img_pp, (2, 3, 1)); # WHC
            temp_img_pp = Flux.unsqueeze(temp_img_pp, 4); # W.H.C.N
            loaded_px = cat(temp_img_pp, loaded_px; dims=4);
          
            global dispatch_indx_record += 1;
            
        end
    end
    
    imgdata_loader = Flux.Data.DataLoader(loaded_img;
     batchsize=dispatch_size, shuffle=shuffle_enable);

    labelbbox_loader = Flux.Data.DataLoader(loaded_bbox;
    batchsize=dispatch_size, shuffle=shuffle_enable);

    labelpx_loader = Flux.Data.DataLoader(loaded_px;
    batchsize=dispatch_size, shuffle=shuffle_enable);

    return imgdata_loader, labelbbox_loader, labelpx_loader;
end

function resetDispatchRecord()
    global dispatch_indx_record = 1;
end

function getDispatchRecord()
    return dispatch_indx_record;
end

function showImageSample(sampleImage::Array{T,3}, bbox::Matrix{S}) where {T <: Real,S <: Real}
    xb, yb, wb, hb = bbox;
    figplot = ImageView.imshow(sampleImage);
    ImageView.annotate!(figplot,
    ImageView.AnnotationBox(xb, yb, xb + wb, yb + hb, linewidth=2, color=Images.RGB(0, 1, 0))
    ); # left top right bottom
end

