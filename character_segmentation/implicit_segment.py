#import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import asarray
import os

class CharacterSegmentation2:
    def __init__(self, path, resultPath):
        self.path = path
        self.img = cv2.imread(path)
        self.threshold = round(self.img.shape[0] // 18)
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
        ####oversegment at h/18
        currentPosition = 0
        w = self.img.shape[1]
        PSCs = []
        end = False
        while(currentPosition <= self.img.shape[1]):
            PSCs.append(currentPosition)
            if currentPosition == self.img.shape[1]:
                end = True
            currentPosition += self.threshold
        if not end:
            PSCs.append(self.img.shape[1])
        plt.imshow(thinned)
        for PSC in PSCs:
            plt.axvline(x=PSC , color='red')
        plt.show()

        
        ####Loop determination


        hist = np.sum(pixelArr, axis=0)
        loop_removed_PSCs = []
        for i in PSCs:
            if hist[i] <= 1:
                loop_removed_PSCs.append(i)
        plt.imshow(thinned)
        for PSC in loop_removed_PSCs:
            plt.axvline(x=PSC , color='red')
        plt.show()

        ####Character boundaries


        # Merge PSCs into segmentation columns (SCs)
        groupedPSCs = [[0]]  
        for i in range(len(loop_removed_PSCs)):
            if i == 0:
                continue
            if loop_removed_PSCs[i] - loop_removed_PSCs[i-1] <= self.threshold:
                groupedPSCs[-1].append(loop_removed_PSCs[i])
            else:
                groupedPSCs.append([loop_removed_PSCs[i]])
        print(groupedPSCs)
        plt.imshow(thinned)
        for PSC in groupedPSCs:
            for element in PSC:
                plt.axvline(x=element , color='red')
        plt.show()
        bound_PSCs = []
        for group in groupedPSCs:
            group = [group[0], group[-1]]
            if group[-1] - group[0] <= 25:
                group = [round((group[-1] + group[0])/2)]
            bound_PSCs.append(group)
        print(bound_PSCs)
        plt.imshow(thinned)
        SCs = []
        for pair in bound_PSCs:
            for PSC in pair:
                SCs.append(PSC)
        
        plt.imshow(thinned)
        for SC in SCs:
            plt.axvline(x=SC , color='red')
        plt.show()

        return SCs
        

    def segmentCharacters(self, SCs):
        charImgList = []
        for i in range(len(SCs) - 1):
            charImg = self.img[0:, SCs[i] : SCs[i+1]]
            charImgList.append(charImg)
        return charImgList

    def segment(self, charImgList):
        counter = 0
        print(len(charImgList))
        for charImg in charImgList:
            charPath = os.path.join(self.resultPath, str(counter) +".png")
            print(charPath)
            cv2.imwrite(charPath, charImg)
            counter = counter + 1
       
    def run(self):
        thinned, pixelArr = self.preprocess()
        SCs = self.findSegCol(thinned, pixelArr)
        charImgList = self.segmentCharacters(SCs)
        self.segment(charImgList) 


