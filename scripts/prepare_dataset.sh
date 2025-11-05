NAME=$1 # name of the dataset in the data folder, e.g., ESCC
WSI=data/${NAME}/images
PROMPT=data/${NAME}/anno
LABEL=data/${NAME}/label.txt
OUT=data/${NAME}/patch
WORKER_NUM=10
PATCH_SIZE=256

# slice wsi to patches via vips
python prepare/wsi_to_patch.py ${WSI} ${OUT}/images ${WORKER_NUM} ${PATCH_SIZE}

# concert xml label to png ground truth
python prepare/xml_to_gt.py ${PROMPT} ${OUT}/gt/ ${WSI} ${PATCH_SIZE}

# make the dateset info, uncommand it for a new dateset (already provided)
python prepare/make_data_info.py ${WSI} ${OUT}/gt/ ${LABEL} data_info/${NAME}.json
