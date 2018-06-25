#!/usr/bin/env bash
source ../conf/config.shlib

OBJECT_DETECTION_PATH=$(config_get OBJECT_DETECTION_PATH)
TRAINING_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get TRAINING_FOLDER)
CONFIG_FILE=$(config_get CONFIG_FILE)

python3 ${OBJECT_DETECTION_PATH}/train.py --logtostderr --train_dir=${TRAINING_PATH}/ --pipeline_config_path=${TRAINING_PATH}/${CONFIG_FILE} &
