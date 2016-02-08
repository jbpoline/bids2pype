#!bash

echo `pwd`
bidsdata='/home/jb/data/bids'
echo "cp ds-005*_model.json "$bidsdata"/ds005/models"
cp ds-005_*model.json $bidsdata"/ds005/models"
echo "cp ds-105_level-all_model.json" $bidsdata"/ds105/models"
cp ds-105_level-all_model.json $bidsdata"/ds105/models"
echo "cp ds-105_level-run_model.json" $bidsdata"/ds105/models"
cp ds-105_level-run_model.json $bidsdata"/ds105/models"
echo "ds-105_level-run_sub-2_model.json" $bidsdata"/ds005/sub-2/models"
cp ds-105_level-run_sub-2_model.json $bidsdata"/ds105/sub-2/func/models"


# /home/jb/code/bids2pype/bids2pype/examples_models_n_spec/models/ds-105_level-run_sub-2_model.json
         
