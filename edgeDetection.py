import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

# First define 1 dimensional helper gaussian
def oneDimGaussian(x,stdDev):
    return (1/(stdDev*(2*np.pi)**0.5))* np.e**(-(x**2)/(2*stdDev**2))

def getGaussBlur(stdDev):
    # generate size of kernel
    # from centre to 3 stdDevs out the gaussian is approx zero, hence for a = 2*(3*stdDev), size is a x a (a should be odd)
    sizeLength = 2*(3*stdDev) + 1
    ker = np.arange(start=-(sizeLength//2),stop=sizeLength//2+1,step=1,dtype=float)
    # assuming the gaussian is centred (mean of (0,0))
    gz = np.array(
        [oneDimGaussian(x,stdDev) for x in ker]
    )
    # reshape to not have trailing ',' can get errors
    if gz.shape != (len(ker),1):
        gz = gz.reshape((len(ker),1))
        assert(gz.shape == (len(ker),1))
    
    # compute outer product to get matrix of shape sideLength x sideLength
    kernel = np.outer(gz,gz.T)
    kernel = kernel/kernel.max()

    # Testing
    # plt.imshow(kernel,interpolation=None,cmap='gray')
    # plt.show()

    return kernel
    
def convolution(kernel,im):
    # get bounds
    imWidth = im.shape[1]; imHeight = im.shape[0]
    kernelWidth = kernel.shape[1]; kernelHeight = kernel.shape[0]
    # place holder for gradient vals
    conv = np.zeros((imHeight,imWidth))

    # need to add rows and cols to image in case the mask goes off of the real image
    # need to +/- by 1 because dim of kernel are odd
    extraCols = int((kernelWidth+1)/2) # along horizontal <---->
    extraRows = int((kernelHeight+1)/2) # along vertical

    imToConv = np.zeros((imHeight + 2*extraCols,imWidth + 2*extraRows))

    # same image with zeros surrounding it
    for col in range(extraCols,imWidth + extraCols): # along x axis
        for row in range(extraRows,imHeight + extraRows): # along y axis
            # print(col,row)
            imToConv[row,col] = im[row-extraRows,col-extraCols]
    # do masking
    for row in range(0,imHeight,1):
        for col in range(0,imWidth,1):
            imgVals = imToConv[row:row+kernelHeight,col:col+kernelWidth]
            conv[row][col] = (np.multiply(kernel,imgVals)).sum()
    # weighted avg
    conv = conv/(kernelWidth*kernelHeight)
    return conv
    
def getGradientMagnitude(gx,gy):
    rval = (gx**2 + gy**2)**0.5
    return rval

def getThreshold(img):
    imHeight = img.shape[0]; imWidth = img.shape[1]
    t = np.sum(img) / (imHeight*imWidth)

    
    shape = img.shape 
    # flatten img to make things easier (not keeping track of idx)
    img = img.flatten()
    cnt = 0

    satisfied = False
    while not satisfied:
        # get lower and upper classes
        lowerClass = [img[i] for i in (np.where(img<t)[0])]
        upperClass = [img[i] for i in (np.where(img>=t)[0])]
        # compute avgs
        lowerAvg = sum(lowerClass)/len(lowerClass); upperAvg = sum(upperClass)/len(upperClass)
        
        
        # print('Lower Avg: ',lowerAvg)
        # print('Upper Avg: ',upperAvg)
        
        # update threshold
        tUpdated = 0.5*(lowerAvg+upperAvg)
        # check 
        
        
        if (abs(t-tUpdated)) <= 0.0001: # good enough threshold difference
            satisfied = True

        # # condition not met --> update t-value
        t = tUpdated
        img = np.where(img>=t,(img*0)+255,img*0)
        

    # reshape image back to old shape
    img = np.reshape(img,shape)
    
    return img

gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def mainED(gx=gx,gy=gy):
    # get image and convert to grayscale
    path = None
    while path is None:
        try:
            path = input('Please Provide Input Image Path')
        except FileNotFoundError:
            print('Incorrect Path! Could not find file, try again. ')
            
    # path = input('Please Provide Input Image Path')
    pic = (Image.open(path)).convert(mode='L')
    pixels = np.asarray(pic)

    stdDev = 1
    # get gauss kernel
    gauss = getGaussBlur(stdDev)
    gaussPix = convolution(gauss,pixels)

    #get grads 
    gradX = convolution(gx,gaussPix); gradY = convolution(gy,gaussPix)

    #get entire pic grad
    pixelsGrad = getGradientMagnitude(gradX,gradY)

    # compute threshold
    finalImgBW = Image.fromarray(
        getThreshold(pixelsGrad)
        ).convert(mode='L')
    

    finalImgBW.show()
    finalImgBW.save(path[0:len(path)-4]  +'EdgeDetection.png')
