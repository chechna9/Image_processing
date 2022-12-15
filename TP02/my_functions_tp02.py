import tools
import matplotlib.pyplot as plt

def get_gaussian_filtre(dimension=3,sigma=0.5):
    """return the result after apply a gaussian filtre with a given dimension """
    kernel = tools.gaussian_mask(size=dimension,sigma=sigma)
    return kernel
def filter_analysis(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
    dimK = kernel.shape[0]
    filtred_im = tools.Conv2D(tools.add_padding(img,((dimK-1)//2,(dimK-1)//2)),kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')
def hello():
    print("hello world")