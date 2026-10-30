# purpose: sort the dataset into categories based on their expression
# create folders in OpenEmotion:
# 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear,
# 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
# step 1: retrieve image
#   A. get image file name from images folder
#   B. read number value + attach exp.npy to get corresponding annotations file (n.jpg -> n_exp.npy)
# step 2: classify image
#   A. find path to said annotations file
#   B. read data in file
#   C. correspond to folder with correct label
# step 3: save image in folder
# repeat for 100K images in training and around 5K (all) in testing

# path and filename

import os
import numpy
import shutil

EmotionCategories = {
    # 0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
    # 8: "None",
    # 9: "Uncertain",
    # 10: "No-Face"
}

TrainingPath = "../AffectNetData/Train/"
TestingPath = "../AffectNetData/Test/"
# testing
imagelist = os.listdir("../AffectNet_ALL/val_set/images")
AnnotationsTestPath = "../AffectNet_ALL/val_set/annotations/"
# training
imagetraininglist = os.listdir("../AffectNet_ALL/train_set/images")
AnnotationsTrainPath = "../AffectNet_ALL/train_set/annotations"


def createfolders():
    for dicitem in EmotionCategories:
        print(EmotionCategories[dicitem],type(EmotionCategories[dicitem]))
        CategoryTrainPath = os.path.join(TrainingPath,EmotionCategories[dicitem]) # path for given emotion
        CategoryTestPath = os.path.join(TestingPath,EmotionCategories[dicitem])
        if os.path.exists(CategoryTrainPath):
            print(CategoryTrainPath, "category already exists")
        else:
            os.makedirs(CategoryTrainPath) # makes the folder for emotion based on its path
        if os.path.exists(CategoryTestPath):
            print(CategoryTestPath, "category already exists")
        else:
            os.makedirs(CategoryTestPath)


def sorttesting():
    for filename in imagelist:
        print(filename, " ------- ", os.path.splitext(filename)[0] + "_exp.npy")
        Anno = os.path.join(AnnotationsTestPath, os.path.splitext(filename)[0] + "_exp.npy") # path for exp file
        AnnoCat = int(numpy.load(Anno).item()) # loads file + retrieves category as int
        if AnnoCat not in [0,8,9,10]:
            CorrectFolderPath = os.path.join(TestingPath, EmotionCategories[AnnoCat]) # path to correct folder
            SourceFileNameWithPath = os.path.join("../AffectNet_ALL/val_set/images",filename)
            DestinationFileNameWithPath = os.path.join(CorrectFolderPath,filename)
            shutil.copy(SourceFileNameWithPath, DestinationFileNameWithPath)

def sorttraining():
    counter = 0
    for filename in imagetraininglist:
        if counter > 100000:
            break
        print(filename, " ------- ", os.path.splitext(filename)[0] + "_exp.npy")
        Anno = os.path.join(AnnotationsTrainPath, os.path.splitext(filename)[0] + "_exp.npy") 
        AnnoCat = int(numpy.load(Anno).item()) 
        if AnnoCat not in [0, 8, 9, 10]:
            CorrectFolderPath = os.path.join(TrainingPath, EmotionCategories[AnnoCat])
            SourceFileNameWithPath = os.path.join("../AffectNet_ALL/train_set/images",filename) 
            DestinationFileNameWithPath = os.path.join(CorrectFolderPath,filename)
            shutil.copy(SourceFileNameWithPath, DestinationFileNameWithPath)
        counter += 1

def main():
    createfolders()
    # sorttesting()
    # sorttraining()

if __name__ == "__main__":
    main()
