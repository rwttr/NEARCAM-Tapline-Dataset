# NEARCAM Tapline Dataset
 Julia API for Near-Range RGB-D Images of Rubber Tapping Line  

## Intend Use within REPL
```julia
include("NearcamTaplineDataset.jl)
import .NearcamTaplineDataset as nds
```
## Data are arranged in K-Fold (k=5) 
single fold option also is avaiable
```julia
# build/initialize dataset
nds.init() #default dataset path
nds.init(datapath = "full path to dataset", kfold_enable=false) # custom path with single fold option

# accessing data
nds.datafold_k.training_data # k= 1,2,3,4,5, single_fold mode k=1
nds.datafold_k.testing_data
```
## Dispatching Data
```julia
nds.dispatchData(dataFold_k.traning_data)
```
