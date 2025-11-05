NAME=$1 # name of the dataset in the data folder, e.g., ESCC
WSI=data/${NAME}/images
PROMPT=data/${NAME}/annotations
OUT=data/${NAME}/patch
WORKER_NUM=5
PATCH_SIZE=256

# slice wsi to patches via vips
python prepare/wsi_to_patch.py ${WSI} ${OUT}/images ${WORKER_NUM} ${PATCH_SIZE}

# concert xml label to png ground truth 
python prepare/xml_to_gt_camelyon.py ${PROMPT} ${OUT}/gt/ ${WSI} ${PATCH_SIZE}

# make the dateset info, uncommand it for a new dateset (already provided)
if echo "$NAME" | grep -q "17"; then
    python prepare/make_data_info_camelyon17.py ${OUT}/gt/ data_info/${NAME}.json
else
    python prepare/make_data_info_camelyon16.py ${OUT}/gt/ data_info/${NAME}.json
fi
