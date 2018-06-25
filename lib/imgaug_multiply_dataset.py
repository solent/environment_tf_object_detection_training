import os, sys, shutil
import imgaug as ia
from imgaug import augmenters as iaa
import glob
from scipy import ndimage
from scipy import misc
import re
from shutil import copyfile
import numpy as np
import time

IMAGES_FOLDER = "/home/alabaere/Images/vespa_velutina_imgaug/"
FOLDER_SEPARATOR = "/"
IMAGES_SIZE = 300
IMAGES_NUMBER_BY_TRANSFORMATION = 1 
#ROTATIONS = [90, 180, 270]
ROTATIONS = [90]

# Méthode utilitaire remplaçant une chaîne de caractère par un autre dans un fichier
def replaceStringInFile(findStr, repStr, filePath):
    tempName = filePath + "~~"
    backupName = filePath + "~"

    inputFile = open(filePath)
    outputFile = open(tempName, "w")
  
    textContent = inputFile.read()
    outputFile.write(textContent.replace(findStr, repStr))

    outputFile.close()
    inputFile.close()

    shutil.copy2(filePath, backupName)
    os.remove(filePath)
    os.rename(tempName, filePath)
    
    os.remove(backupName)

# Méthode de récupération des transformations à effectuer sur les images
def getIaaSequential(rotation):
    return iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            rotate=rotation,
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            shear=(-8, 8)
        )
    ], random_order=True)

# Méthode de récupération du nom du fichier xml associé à une image
def getXmlFilePathFromImage(imageFilePath):
    return "{}{}{}.xml".format(imageFilePath[0:imageFilePath.rfind(FOLDER_SEPARATOR)], FOLDER_SEPARATOR, imageFilePath[imageFilePath.rfind(FOLDER_SEPARATOR) + 1:-4])

# Méthode de récupération du nom du nouveau fichier xml associé à une image
def getNewXmlFilePathFromImage(xmlFilePath, rotationIndex, imageIndex):
    return "{}_{}_{}.xml".format(xmlFilePath[:-4], rotationIndex, imageIndex)

# Méthode de récupération du nom d'une nouvelle image associée à une image
def getNewImageFilePathFromImage(imageFilePath, rotationIndex, imageIndex):
    return "{}_{}_{}.jpg".format(imageFilePath[:-4], rotationIndex, imageIndex)
    
# Méthode de récupération des "bound boxes" d'un fichier xml
def getBndBoxes(xmlFilePath):
    xmlFile = open(xmlFilePath, "r")
    # Recherche de toutes les "bound boxes"
    bndboxs = re.findall(r'<bndbox>(.*?)</bndbox>', xmlFile.read(), re.MULTILINE|re.DOTALL)
    # Fermeture du fichier
    xmlFile.close()

    return bndboxs
    
# Méthode de récupération des keypoints associés aux "bound boxes" d'un fichier xml
def getBndBoxesKeyPoints(bndboxes):
    keypoints = []

    for bndbox in bndboxs:
        keypoints += getBndBoxKeyPoints(bndbox)

    return keypoints

# Méthode de récupération des keypoints associés à une "bound box" d'un fichier xml
def getBndBoxKeyPoints(bndbox):
    keypoints = []
    
    xmin = re.search(r'<xmin>(.*?)</xmin>', bndbox, re.MULTILINE|re.DOTALL).group(1)
    xmax = re.search(r'<xmax>(.*?)</xmax>', bndbox, re.MULTILINE|re.DOTALL).group(1)
    ymin = re.search(r'<ymin>(.*?)</ymin>', bndbox, re.MULTILINE|re.DOTALL).group(1)
    ymax = re.search(r'<ymax>(.*?)</ymax>', bndbox, re.MULTILINE|re.DOTALL).group(1)
    #print("xmin : " + xmin + ", xmax : " + xmax + ", ymin : " + ymin + ", ymax : " + ymax)

    m1 = ia.Keypoint(x=int(xmin), y=int(ymin))
    keypoints.append(m1)
    m2 = ia.Keypoint(x=int(xmin), y=int(ymax))
    keypoints.append(m2)
    m3 = ia.Keypoint(x=int(xmax), y=int(ymin))
    keypoints.append(m3)
    m4 = ia.Keypoint(x=int(xmax), y=int(ymax))
    keypoints.append(m4)

    return keypoints

# Méthode de récupération, après transformation, de la nouvelle "bound box" d'un fichier xml
def controlKeypointsValidity(keypoints):
    hasValidKeypoints = True
    
    increment = 0
    keypointsAugLength = len(keypointsAug)
    while hasValidKeypoints == True and increment < keypointsAugLength:
        if float(keypointsAug[increment].x) < float(0) or float(keypointsAug[increment].x) > float(IMAGES_SIZE) or float(keypointsAug[increment].y) < float(0) or float(keypointsAug[increment].y) > float(IMAGES_SIZE):
            hasValidKeypoints = False
        increment = increment + 1

    return hasValidKeypoints

# Méthode réécrivant, après transformation, les "bound boxes" d'un fichier xml
def rewriteBndBoxesIntoNewXmlFile(keypointsAug, xmlFilePath, newXmlFilePath):
    # Copie du fichier xml dans un nouveau
    copyfile(xmlFilePath, newXmlFilePath)
    
    # Réouverture du fichier xml associé à l'image
    xmlFile = open(xmlFilePath, "r")
    # Recherche de toutes les "bound boxes"
    bndboxs = re.findall(r'<bndbox>(.*?)</bndbox>', xmlFile.read(), re.MULTILINE|re.DOTALL)
    # Réécriture des points associés aux "bound boxes"
    for index, bndbox in enumerate(bndboxs):
        newM1 = keypointsAug[index]
        newM2 = keypointsAug[index + 1]
        newM3 = keypointsAug[index + 2]
        newM4 = keypointsAug[index + 3]
        
        #print("new M1 ({}, {})".format(int(newM1.x), int(newM1.y)))
        #print("new M2 ({}, {})".format(int(newM2.x), int(newM2.y)))
        #print("new M3 ({}, {})".format(int(newM3.x), int(newM3.y)))
        #print("new M4 ({}, {})".format(int(newM4.x), int(newM4.y)))

        newBndbox = getNewBndBox(bndbox, newM1, newM2, newM3, newM4)
        
        replaceStringInFile(bndbox, newBndbox, newXmlFilePath)

    xmlFile.close()
        
# Méthode de récupération, après transformation, de la nouvelle "bound box" d'un fichier xml
def getNewBndBox(bndbox, newM1, newM2, newM3, newM4):
    # On considère une rotation à ANGLE DROIT
    newXmin = min(int(newM1.x), int(newM2.x), int(newM3.x), int(newM4.x))
    newYmin = min(int(newM1.y), int(newM2.y), int(newM3.y), int(newM4.y))
    newXmax = max(int(newM1.x), int(newM2.x), int(newM3.x), int(newM4.x))
    newYmax = max(int(newM1.y), int(newM2.y), int(newM3.y), int(newM4.y))

    newBndbox = bndbox
    newBndbox = re.sub(r'<xmin>(.*?)</xmin>', "<xmin>{}</xmin>".format(newXmin), newBndbox, re.MULTILINE|re.DOTALL)
    newBndbox = re.sub(r'<ymin>(.*?)</ymin>', "<ymin>{}</ymin>".format(newYmin), newBndbox, re.MULTILINE|re.DOTALL)
    newBndbox = re.sub(r'<xmax>(.*?)</xmax>', "<xmax>{}</xmax>".format(newXmax), newBndbox, re.MULTILINE|re.DOTALL)
    newBndbox = re.sub(r'<ymax>(.*?)</ymax>', "<ymax>{}</ymax>".format(newYmax), newBndbox, re.MULTILINE|re.DOTALL)

    return newBndbox

# Récupération des chemins des images du dossier
imageFilePathsIntoFolder = glob.glob(IMAGES_FOLDER + "*.jpg")

# Définition des compteurs
treatedFileCounter = 0
fileToBeTreatedNumber = len(imageFilePathsIntoFolder)
newImagesCounter = 0
errorCounter = 0

# Boucle sur tous les fichiers .jpg du dossier
for imageFilePath in imageFilePathsIntoFolder:
    start = time.time()
    
    # Boucle sur les différentes rotations
    for rotationIndex, rotation in enumerate(ROTATIONS):
        # Récupération du nom du fichier xml associé à l'image
        xmlFilePath = getXmlFilePathFromImage(imageFilePath)
        # Recherche de toutes les "bound boxes"
        bndboxs = getBndBoxes(xmlFilePath)
        # Récupération des keypoints associés aux "bound boxes"
        keypoints = getBndBoxesKeyPoints(bndboxs)

        # L'image est dupliquée autant de fois qu'elle doit subir de transformations différentes
        images = np.array(
            [ndimage.imread(imageFilePath, mode="RGB") for _ in range(IMAGES_NUMBER_BY_TRANSFORMATION)],
            dtype=np.uint8
        )
        # De même pour les keypoints
        keypointsOnImage = []
        for imageIndex in range(IMAGES_NUMBER_BY_TRANSFORMATION):
            keypointsOnImage.append(ia.KeypointsOnImage(keypoints, shape=images[imageIndex].shape))

        # Récupération des transformations à effectuer sur les images
        seq = getIaaSequential(rotation)

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the keypoints and it will
        # lead to the same augmentations.
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
        # exactly same augmentations for every batch!
        seq_det = seq.to_deterministic()

        # Transformation des images et keypoints
        imagesAug = seq_det.augment_images(images)
        keypointsOnImageAug = seq_det.augment_keypoints(keypointsOnImage)

        for imageIndex in range(IMAGES_NUMBER_BY_TRANSFORMATION):
            keypointsAug = keypointsOnImageAug[imageIndex].keypoints
            hasValidKeypoints = controlKeypointsValidity(keypointsAug)

            if hasValidKeypoints == True:            
                # Génération de la nouvelle image .jpg
                misc.imsave(getNewImageFilePathFromImage(imageFilePath, rotationIndex, imageIndex), imagesAug[imageIndex])            
                #misc.imsave(getNewImageFilePathFromImage(imageFilePath, rotationIndex, imageIndex), keypointsOnImageAug[imageIndex].draw_on_image(imagesAug[imageIndex], size=7))

                # Génération du fichier xml associé à la nouvelle image
                newXmlFilePath = getNewXmlFilePathFromImage(xmlFilePath, rotationIndex, imageIndex)
                rewriteBndBoxesIntoNewXmlFile(keypointsOnImageAug[imageIndex].keypoints, xmlFilePath, newXmlFilePath)

                newImagesCounter = newImagesCounter + 1
            else:
                errorCounter = errorCounter + 1

    print(time.time() - start)
    treatedFileCounter = treatedFileCounter + 1
    print('{} fichier(s) traité(s) sur {} en {}s'.format(treatedFileCounter, fileToBeTreatedNumber, time.time() - start))

print('Traitement terminé : {} nouvelles images créées, {} "erreurs"'.format(newImagesCounter, errorCounter))

            
    

    
