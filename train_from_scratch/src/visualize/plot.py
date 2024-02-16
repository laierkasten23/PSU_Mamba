import cv2
import matplotlib.pyplot as plt

def plot_image_with_groundtruth(image, groundtruth):

    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].set_title("Image")
    ax[0].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_title("Mask")
    ax[1].imshow(groundtruth)
    ax[1].set_axis_off()
    plt.show()


