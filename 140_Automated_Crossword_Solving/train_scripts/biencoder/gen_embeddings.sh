OUTFOLDER=$1
CHECKPOINT=$2
WORDLIST=$3


if [ -n "$(find "$OUTFOLDER" -maxdepth 0 -type d -empty 2>/dev/null)" ]; then
    echo "Yay! the folder is empty"
else
    echo "No! folder is not empty, exiting"
    exit
fi

COUNTER=0
DEVICES=(1 2 3 4)
NUM_DEVICES=${#DEVICES[@]}
for DEVICE in ${DEVICES[@]}
do	
python DPR/generate_dense_embeddings.py \
	--shard_id $COUNTER --num_shards $NUM_DEVICES \
	--batch_size 4000 \
	--model_file $CHECKPOINT \
	--out_file $OUTFOLDER/embeddings.json \
	--ctx_file $WORDLIST & 

COUNTER=$((COUNTER + 1))
done	
