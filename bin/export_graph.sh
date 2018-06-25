#!/usr/bin/env bash
source ../conf/config.shlib

OBJECT_DETECTION_PATH=$(config_get OBJECT_DETECTION_PATH)
TRAINING_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get TRAINING_FOLDER)
GRAPH_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get GRAPH_FOLDER)
CONFIG_FILE=$(config_get CONFIG_FILE)

modelFullPath=$(ls ${TRAINING_PATH}/model.ckpt-*.index | tail -n -1)
modelFullPathWithoutExt=${modelFullPath:0:-6}
echo $modelFullPathWithoutExt

rm -r ${GRAPH_PATH}/*
python3 ${OBJECT_DETECTION_PATH}/export_inference_graph.py --input_type image_tensor --pipeline_config_path ${TRAINING_PATH}/${CONFIG_FILE} --trained_checkpoint_prefix ${modelFullPathWithoutExt} --output_directory ${GRAPH_PATH}
