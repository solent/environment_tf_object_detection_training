#!/usr/bin/env bash
source ../conf/config.shlib

IMAGES_SOURCE_GROUPING_PATH=$(config_get IMAGES_SOURCE_PATH)/$(config_get PROJECT_FOLDER)
DATA_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get DATA_FOLDER)
LIB_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get LIB_FOLDER)
TRAIN_RECORD_FILE=$(config_get TRAIN_RECORD_FILE)
TEST_RECORD_FILE=$(config_get TEST_RECORD_FILE)

rm ${DATA_PATH}/${TRAIN_RECORD_FILE}
rm ${DATA_PATH}/${TEST_RECORD_FILE}

python3 ${LIB_PATH}/xml_to_csv.py

python3 ${LIB_PATH}/generate_tfrecord.py --csv_input=${DATA_PATH}/train_labels.csv  --output_path=${DATA_PATH}/${TRAIN_RECORD_FILE} --images_path=${IMAGES_SOURCE_GROUPING_PATH}
python3 ${LIB_PATH}/generate_tfrecord.py --csv_input=${DATA_PATH}/test_labels.csv  --output_path=${DATA_PATH}/${TEST_RECORD_FILE} --images_path=${IMAGES_SOURCE_GROUPING_PATH}

rm ${DATA_PATH}/train_labels.csv
rm ${DATA_PATH}/test_labels.csv

