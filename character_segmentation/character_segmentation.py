#import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import asarray

class CharacterSegmentation2:
    def __init__(self, path, resultPath, threshold=4):

        # Open image
        self.path = path
        self.img = cv2.imread(path)
        self.threshold = threshold
        self.resultPath = resultPath
    def preprocess(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #Expand the image to the left and right with 2 background col
        h = thresh.shape[0]
        w = thresh.shape[1]
        expand = np.zeros((h, w+2), np.uint8)
        expand[0:h, 1 : w + 1] = thresh[0:h, 0:w]

        #Thin the word
        thinned = cv2.ximgproc.thinning(expand, thinningType=cv2.ximgproc.THINNING_GUOHALL)

        #Binarize
        pixelArr = asarray(thinned)
        pixelArr = pixelArr.astype('float32')
        pixelArr /= 255
        return thinned, pixelArr
    def findSegCol(self, thinned, pixelArr):
        hist = np.sum(pixelArr, axis=0)
        PSCs = []
        for i in range(len(hist)):
            if hist[i] == 0 or hist[i] == 1:
                PSCs.append(i)
        plt.imshow(thinned)
        for PSC in PSCs:
            plt.axvline(x=PSC , color='red')
        plt.show()

        # Merge PSCs into segmentation columns (SCs)
        groupedPSCs = [[0]]  
        for i in range(len(PSCs)):
            if i == 0:
                continue
            if PSCs[i] - PSCs[i-1] <= self.threshold:
                groupedPSCs[-1].append(PSCs[i])
            else:
                groupedPSCs.append([PSCs[i]])
        SCs = []
        for i in range(len(groupedPSCs)):
            SCs.append(np.around(np.mean(groupedPSCs[i])).astype(int))
        plt.imshow(thinned)
        for SC in SCs:
            plt.axvline(x=SC , color='red')
        plt.show()
        return SCs
    def segmentCharacters(self, SCs):
        s=0
        charImgList = []
        for i in range(len(SCs) - 1):
            charImg = self.img[0:, SCs[i] : SCs[i+1]]
            charImgList.append(charImg)
        return charImgList

    def segment(self, charImgList):
        counter = 0
        print(len(charImgList))
        for charImg in charImgList:
            charPath = self.resultPath + str(counter) +".jpeg"
            print(charPath)
            cv2.imwrite(charPath, charImg)
            counter = counter + 1
       
    def run(self):
        thinned, pixelArr = self.preprocess()
        SCs = self.findSegCol(thinned, pixelArr)
        charImgList = self.segmentCharacters(SCs)
        self.segment(charImgList) 

    
