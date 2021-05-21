# NEARCAM Tapline Dataset
 Julia API for Near-Range RGB-D Images of Rubber Tapping Line  

##Intend Use within REPL
```
include("NearcamTaplineDataset.jl)
import NearcamTaplineDataset as nds
nds.init() 
```
##Data are arranged in K-Fold (k=5) ps. single fold option is avaiable
```
dataFold_k.training 
dataFold_k.testing 
```
