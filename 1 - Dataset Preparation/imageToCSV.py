from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import cv2

def main():
    columns = []
    for i in range(1, (32*32)+1):
        columns.append('pixel' + str(i))
    columns.append('class')
    arrData =[]
    
    pathG = 'processedData/girls/'
    pathB = 'processedData/boys/'
    imagesG = [f for f in listdir(pathG) if isfile(join(pathG, f))]
    imagesB = [f for f in listdir(pathB) if isfile(join(pathB, f))]

    for image in imagesG:
        path = pathG + image
        img = cv2.imread(path, 0)
        if img.shape == (32,32):
            temp = img.reshape(1024)
            temp = np.append([temp], 'girl')
            arrData.append(temp)
    
    for image in imagesB:
        path = pathB + image
        img = cv2.imread(path, 0)
        if img.shape == (32,32):
            temp = img.reshape(1024)
            temp = np.append([temp], 'boy')
            arrData.append(temp)

    
    data = pd.DataFrame(columns=columns,data=arrData)

    data.to_csv('csvData/faces.csv')

if __name__ == "__main__":
    main()