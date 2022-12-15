import tools
import matplotlib.pyplot as plt
import numpy as np
def get_gaussian_filtre(dimension=3,sigma=0.5):
    """return the result after apply a gaussian filtre with a given dimension """
    kernel = tools.gaussian_mask(size=dimension,sigma=sigma)
    return kernel
def filter_analysis(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
    filtred_im =apply_filter_to_single_channel(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')
def apply_filter_to_single_channel(img,kernel):
    dimK = kernel.shape
    return tools.Conv2D(tools.add_padding(img,((dimK[0]-1)//2,(dimK[1]-1)//2)),kernel)
def apply_filter_to_colored_img(img,kernel):
#     dstack build ndarray on the third axis
    return np.dstack([apply_filter_to_single_channel(img[:,:,z],kernel) for z in range(3)])
def filter_analysis_colored(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
   
    filtred_im = apply_filter_to_colored_img(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')