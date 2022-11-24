#Ce fichier sera alloué aux fonctions de la partie II du TP 01b
import matplotlib.pyplot as plt
import numpy as np
def OpenImage(file):
    """
    return matrix I, C number of cols, L number of lingnes
        """
    I=plt.imread(file)
    L,C = I.shape[:2]
    return (I,L,C)
def Divide(I):
    Red = I[:,:,0]
    Green = I[:,:,1]
    Blue = I[:,:,2]
    return (Red,Green,Blue)
def HSV(file):
    import cv2
    image = cv2.imread(file,3)
    image_HSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return image_HSV
def CountPix(file):
    (I,L,C) = OpenImage(file)
    return L*C
def factPix(a,b,I):
    I_fact =I*a+b
    return I_fact
def func_a(I):
    loga = np.log2(I)
    exp = np.exp(I)
    carre = np.sqrt(I)
    return (loga,exp,carre)
def func_m(I):
    mean = I.mean()
    std = I.std()
    return (mean,std)
def normalize(I):
    (mean,std) = func_m(I)
    I_norm=(I-mean)/std
    return I_norm
def inverse(I):
    I_inv = I.max()-I
    return I_inv
def calcHist(I):
    H,b=np.histogram(I,bins=256)
    return (H,b)
def thresHold(I,seuil):
    x,y,z = I.shape
    T = I.reshape(x*y*z)
    T = np.array(list(map(lambda x : 0 if x<= seuil else 255 ,T)))  
    return T.reshape(x,y,z)
def func_j(file):
    (I,L,C) = OpenImage(file)
    plt.figure(1)
#     show the image
    plt.imshow(I)
#     calcule the historgram
    I_hist,I_bins = calcHist(I)
#     plot the histrograme 
    plt.figure(2)
    plt.plot(I_hist)
    #inverse
    I_inverse = inverse(I)
    plt.figure(3)
    plt.imshow(I_inverse)
#   plot the histrograme after inversion 
    plt.figure(4)
    I_inverse_hist,I_inverse_bins = calcHist(I_inverse)
    plt.plot(I_inverse_hist)
    return 0
def func_t(file):
    (I,L,C) = OpenImage(file)
    plt.figure(1)
#     show the image
    plt.imshow(I)
#     calcule the historgram
    I_hist,I_bins = calcHist(I)
#     plot the histrograme 
    plt.figure(2)
    plt.plot(I_hist)
    #Normalize
    I_normalize = normalize(I)
    plt.figure(3)
    plt.imshow(I_normalize)
#   plot the histrograme after normalization 
    plt.figure(4)
    I_normalize_hist,I_normalize_bins = calcHist(I_normalize)
    plt.plot(I_normalize_hist)
    return 0
def func_f(file):
    (I,L,C) = OpenImage(file)
    plt.figure(1)
#     show the image
    plt.imshow(I)
#     calcule the historgram
    I_hist,I_bins = calcHist(I)
#     plot the histrograme 
    plt.figure(2)
    plt.plot(I_hist)
    #seuilage
    I_seuile = thresHold(I,128)
    plt.figure(3)
    plt.imshow(I_seuile)
#   plot the histrograme after normalization 
    plt.figure(4)
    I_seuile_hist,I_seuile_bins = calcHist(I_seuile)
    plt.plot(I_seuile_hist)
    return 0
