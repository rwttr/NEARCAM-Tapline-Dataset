# NEARCAM Tapline Dataset
 Julia API for Near-Range RGB-D Images of Rubber Tapping Line  

## Intend Use within REPL
```julia
include("NearcamTaplineDataset.jl)
import .NearcamTaplineDataset as nds
nds.init() # default dataset_path
nds.init(datapath = "full path to dataset", kfold_enable=false) # custom dataset_path with single fold option
```
## Data are arranged in K-Fold (k=5) 
single fold option also is avaiable
```julia
nds.datafold_k.training_data # k= 1,2,3,4,5
nds.datafold_k.testing_data # k= 1,2,3,4,5
```
## Dispatching Datafold
```julia
nds.dispatchData(dataFold_k.traning_data)
```
