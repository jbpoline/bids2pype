#!bash

echo `pwd`
bidsdata='/home/jb/data/bids'
echo "cp ds-005_model.json "$bidsdata"/ds005/models"
cp ds-005_model.json $bidsdata"/ds005/models"
echo "cp ds-105_level-all_model.json" $bidsdata"/ds105/models"
cp ds-105_level-all_model.json $bidsdata"/ds105/models"
echo "ds-105_sub-2_model.json" $bidsdata"/ds005/sub-2/models"
cp ds-105_sub-2_model.json $bidsdata"/ds105/sub-2/models"

