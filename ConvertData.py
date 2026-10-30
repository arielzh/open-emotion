import cv2
import os

TrainingPathOrig = "../AffectNetData/Train/"
TestingPathOrig = "../AffectNetData/Test/"

TrainingPathResized = "../AffectNetDataResized/Train/"
TestingPathResized = "../AffectNetDataResized/Test/"

def makedirs(path):
    if os.path.exists(path):
        print(path, "category already exists")
    else:
        os.makedirs(path)

def convertData(PathOrigFromTopLevel, PathResizedFromTopLevel):
    DirlistOrig = os.listdir(PathOrigFromTopLevel)
    for dirItem in DirlistOrig:
        OrigPath = os.path.join(PathOrigFromTopLevel,dirItem)
        if os.path.isdir(OrigPath):
            DestPath = os.path.join(PathResizedFromTopLevel,dirItem)
            print(DestPath)
            makedirs(DestPath)
            FilelistOrig = os.listdir(OrigPath)
            for fileItem in FilelistOrig:
                OrigFilename = os.path.join(OrigPath,fileItem)
                if os.path.isfile(OrigFilename) and os.path.splitext(OrigFilename)[1] == ".jpg":
                    OrigImg = cv2.imread(OrigFilename)
                    DestImg = cv2.resize(src=OrigImg,dsize=(48,48),interpolation=cv2.INTER_LANCZOS4)
                    DestImg = cv2.cvtColor(DestImg, cv2.COLOR_RGB2GRAY)
                    DestFilename = os.path.join(DestPath,fileItem)
                    # print(DestFilename)
                    cv2.imwrite(DestFilename, DestImg)

def main():
    convertData(TrainingPathOrig,TrainingPathResized) #converts affectnet original to affectnet resized
    convertData(TestingPathOrig,TestingPathResized)


if __name__ == "__main__":
    main()