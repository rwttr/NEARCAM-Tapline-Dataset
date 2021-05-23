# NEARCAM Tapline Dataset
 Julia API for Near-Range RGB-D Rubber Tapping Line Images 

## Intend Use within REPL
```julia
include("NearcamTaplineDataset.jl")
import .NearcamTaplineDataset as nds
```
## Data are arranged in K-Fold (k=5) 
single fold option also avaiable
```julia
# build/initialize dataset
nds.init() #default dataset path
# accessing data
nds.datafold_k.training_data # k= 1,2,3,4,5
nds.datafold_k.test_data

nds.init(datapath = "full path to dataset", kfold_enable=false) # custom path with single fold option
nds.datafold_single.training_data
nds.datafold_single.test_data
```
## Dispatching Data
```julia
# default option
nds.dispatchData(dataFold_k.traning_data)

# custom option
nds.dispatchData(dataFold_k.traning_data; 
    shuffle_enable=false,
    dispatch_size=1, 
    data_selector="rgb1", 
    img_outputsize=[128 128])
```
