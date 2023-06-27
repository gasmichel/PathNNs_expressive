wget https://www.chrsmrrs.com/graphkerneldatasets/DD.zip
wget https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip
wget https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip
wget https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip
wget https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip
wget https://www.chrsmrrs.com/graphkerneldatasets/IMDB-MULTI.zip


wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/DD_splits.json
wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/ENZYMES_splits.json
wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/NCI1_splits.json
wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/PROTEINS_full_splits.json
wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-BINARY_splits.json
wget https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-MULTI_splits.json

mv PROTEINS_full_splits.json PROTEINS_splits.json

for file in *.zip ; 
    do unzip $file ;
    rm $file ;  
done ;

DATASETS="DD ENZYMES PROTEINS NCI1 IMDB-BINARY IMDB-MULTI"
JSON="_splits.json" 

for DATASET in $DATASETS ; 
    do mv "$DATASET$JSON" "$DATASET/$DATASET$JSON" ; 
done ;