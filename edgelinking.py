import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


#-------Line Detection-------#

# Apply gradient mask on image for edge detection, given image and mask
def gradientFilter(image, mask):
    img_list = []
    for img_row in range(int(len(mask)/2), len(image)-int(len(mask)/2)):
        for img_col in range(int(len(mask[0])/2), len(image[0])-int(len(mask[0])/2)):
            img_list.append(np.mean(np.multiply(image[img_row-int(len(mask)/2):img_row+int(len(mask)/2)+1,
                                         img_col-int(len(mask[0])/2):img_col+int(len(mask[0])/2)+1]
                                   , mask)))
    return np.array(img_list).reshape(-1,len(image[0])-len(mask[0])+1)

image = cv2.imread('sudoku.png')
binImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Laplacian mask for edge detection
laplacian = -np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
imageGrad = gradientFilter(binImage, laplacian)

# Creating thresholded binary image of edges detected
maxGradT = np.max(imageGrad)*0.1
imageGrad = np.array([255 if imageGrad[i,j]> maxGradT else 0 for i in range(len(imageGrad)) for j in range(len(imageGrad[0]))]).reshape(-1, len(imageGrad[0]))
imageGrad = np.pad(imageGrad, int(len(laplacian)/2),'edge')

# Calculating rho for a range of theta and the input pixel coordinates
imageGradX, imageGradY = np.where(imageGrad == 255)
thetaRange = np.transpose(np.array([i for i in range(-90,90)]).reshape(-1,1))
rhoCalc = np.round(np.matmul(imageGradX.reshape(-1,1), np.cos((thetaRange*np.pi/180)))+np.matmul(imageGradY.reshape(-1,1), np.sin((thetaRange*np.pi/180))))

# Calculation of hough space accumulator, in matrix - rhoTheta
rhoTheta = []
for i in range(np.int32(np.min(rhoCalc)),np.int32(np.max(rhoCalc)+1)):
    for j in range(thetaRange.size):
        rhoTheta.append(np.size(np.where(rhoCalc[:,j] == i)))
rhoTheta = np.array(rhoTheta).reshape(-1,thetaRange.size)

# Get only lines with atlease 120 hits in the accumulator cells
goodRho, goodTheta = np.where(rhoTheta>120)

imageResult = image.copy()
edges = np.zeros(imageGrad.shape)
indent = np.min(rhoCalc)
Lines = set({})

# Plotting the detected lines on the image
for i,j in zip(goodRho, goodTheta):
    """ Considering lines within a bin of r-12 and r+12 and theta+1 and theta-1 
        to be one line to avoid labelling the same line as multiple lines. """
    ind = np.where((goodRho>=i) & (goodRho<i+25) & (goodTheta<j+4) & (goodTheta>=j))
    if goodRho[ind].size > 0 or goodTheta[ind].size > 0:
        i = np.mean(goodRho[ind])
        j = np.mean(goodTheta[ind])
    else:
        continue
        
    goodRho = np.delete(goodRho, ind)
    goodTheta = np.delete(goodTheta, ind)
    
    xTheta = np.cos((j-90)*np.pi/180)
    yTheta = np.sin((j-90)*np.pi/180)
    
    # Calculate y for each x based on the line equation, knowing r and theta
    for x in range(len(imageGrad)):
        
        # Calculating y value using hough to cartesian line conversion
        y = np.round(((i + indent) - (x*xTheta))/yTheta)
        if y >= len(imageGrad[0]):
            break
        if (y>=0):
            
            # Angle <-85 and >85 used to print only vertical lines
            if (j-90)<-85 or (j-90)>85 or (j-180)>-175 and (j-180)<-175:
                Lines.add((i,j))
                # line thinckness of +/- 2 used for visibility on image
                imageResult[x,int(y)-2:int(y)+2] = [0,255,0]
                edges[x,int(y)-3:int(y)+3] = 255
            

print("Number of lines  : ", len(Lines))
cv2.imwrite('plines.png',imageResult)
