#!/usr/bin/env bash
source ../conf/config.shlib

CLASSES=$(config_get CLASSES)

IMAGES_TRAIN_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get IMAGES_TRAIN_FOLDER)

IMAGES_SOURCE_PATH=$(config_get IMAGES_SOURCE_PATH)
IMAGES_SOURCE_GROUPING_PATH=$(config_get IMAGES_SOURCE_PATH)/$(config_get PROJECT_FOLDER)

IMAGES_TRAIN_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get IMAGES_TRAIN_FOLDER)
IMAGES_TEST_PATH=$(config_get DEV_PATH)/$(config_get PROJECT_FOLDER)/$(config_get IMAGES_TEST_FOLDER)

IMGAUG_SUFFIX=$(config_get IMGAUG_SUFFIX)

# Suppression des images précédentes
rm -f ${IMAGES_TRAIN_PATH}/*
rm -f ${IMAGES_TEST_PATH}/*
rm -f ${IMAGES_SOURCE_GROUPING_PATH}/*

# Copie des images dans des dossiers temporaires avant augmentation des images
for entity in ${CLASSES[@]}
do  
    mkdir ${IMAGES_SOURCE_PATH}/${entity}_${IMGAUG_SUFFIX}
    cp -a ${IMAGES_SOURCE_PATH}/${entity}/. ${IMAGES_SOURCE_PATH}/${entity}_${IMGAUG_SUFFIX}/
done

# TODO : augmentation des images


# Déplacement des images augmentées dans un dossier beekeeper
for entity in ${CLASSES[@]}
do
    mv ${IMAGES_SOURCE_PATH}/${entity}_${IMGAUG_SUFFIX}/* ${IMAGES_SOURCE_GROUPING_PATH}
    rmdir ${IMAGES_SOURCE_PATH}/${entity}_${IMGAUG_SUFFIX}
done

# Copie des images du dossier beekeeper vers le dossier train
cp -a ${IMAGES_SOURCE_GROUPING_PATH}/. ${IMAGES_TRAIN_PATH}/

# Déplacement de 10% des images dans le dossier test
for entity in ${CLASSES[@]}
do
    filesNumber=$(ls ${IMAGES_TRAIN_PATH}/${entity}* | wc -l)
    filesNumberToMove=$(($((filesNumber / (10 * 2))) * 2))
    mv `ls ${IMAGES_TRAIN_PATH}/${entity}* | head -${filesNumberToMove}` ${IMAGES_TEST_PATH}
done
