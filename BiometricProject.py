from tkinter import *
import cv2
from PIL import ImageTk, Image
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog

root1 = Tk()
root1.title("Biometrics Project")
# root.geometry("960x1080")

root1.filename = filedialog.askopenfilename(title='choose file')

photoPath = root1.filename

img = Image.open(photoPath)
img = img.resize((640, 360))
img = ImageTk.PhotoImage(img)


# Functions Block Begining

def loadGreyImage():
    greyImage = cv2.imread(photoPath, cv2.IMREAD_GRAYSCALE)
    greyImages = cv2.resize(greyImage, (1280, 720))
    cv2.imshow('Grey Image', greyImages)


def loadRGBImage():
    RGBImage = cv2.imread(photoPath, cv2.COLOR_BGR2RGB)
    RGBImages = cv2.resize(RGBImage, (1280, 720))
    cv2.imshow('RGB Image', RGBImages)


def loadHSVImage():
    HSVImage = cv2.imread(photoPath, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', HSVImage)


def SplitRGB():
    B, G, R = cv2.split(cv2.imread(photoPath))
    Bs = cv2.resize(B, (1280, 720))
    cv2.imshow('Blue Channel', Bs)
    Gs = cv2.resize(G, (1280, 720))
    cv2.imshow('Green Channel', Gs)
    Rs = cv2.resize(R, (1280, 720))
    cv2.imshow('Red Channel', Rs)


def rotate90():
    image = cv2.imread(photoPath, 0)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    ri = cv2.warpAffine(image, M, (cols, rows))
    ris = cv2.resize(ri, (1280, 720))
    cv2.imshow('Rotated Image', ris)


def morphologyx():
    image = cv2.imread(photoPath, 0)
    (th, bi) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((3, 3), np.uint8)
    oi = cv2.morphologyEx(bi, cv2.MORPH_OPEN, kernal)
    bis = cv2.resize(bi, (1280, 720))
    ois = cv2.resize(oi, (1280, 720))
    cv2.imshow('bi', bis)
    cv2.imshow('OI', ois)


def morphologicalTransformation():
    image = cv2.imread(photoPath, 0)
    (th, bi) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((3, 3), np.uint8)
    oi = cv2.erode(bi, kernal, iterations=1)
    bis = cv2.resize(bi, (1280, 720))
    ois = cv2.resize(oi, (1280, 720))
    cv2.imshow('bi', bis)
    cv2.imshow('OI', ois)


def sobeldervations():
    image = cv2.imread(photoPath)
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobelimag = cv2.add(sobelX, sobelY)
    plt.subplot(141), plt.imshow(image), plt.title('orgignal')
    plt.subplot(142), plt.imshow(sobelX), plt.title('sobelx')
    plt.subplot(143), plt.imshow(sobelY), plt.title('sobely')
    plt.subplot(144), plt.imshow(sobelimag), plt.title('sobel')
    plt.show()


def leplacian():
    img = cv2.imread(photoPath)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    plt.subplot(121), plt.imshow(img), plt.title('original image')
    plt.subplot(122), plt.imshow(laplacian), plt.title('edge image')
    plt.show()


def median():
    originalimage = cv2.imread(photoPath, 0)
    blur = cv2.medianBlur(originalimage, 5)
    plt.subplot(121), plt.imshow(originalimage), plt.title('original')
    plt.subplot(122), plt.imshow(blur), plt.title('blurred')
    plt.show()


def inverse():
    originalimage = cv2.imread(photoPath)
    inverse_image = 255 - originalimage
    inverse_images = cv2.resize(inverse_image, (1280, 720))
    cv2.imshow('inversed', inverse_images)


def bilateral_filtering():
    originalimage = cv2.imread(photoPath, 0)
    blur = cv2.bilateralFilter(originalimage, 9, 9, 75, 75)
    plt.subplot(121), plt.imshow(originalimage), plt.title('original')
    plt.subplot(122), plt.imshow(blur), plt.title('blurred')
    plt.show()


def canny():
    img = cv2.imread(photoPath)
    edge = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.subplot(122), plt.imshow(edge), plt.title('edge_image')
    plt.show()


def dilode():
    originalimage = cv2.imread(photoPath, 0)
    (th, bi) = cv2.threshold(originalimage, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((3, 3), np.uint8)
    oi = cv2.dilate(bi, kernal, iterations=1)
    bis = cv2.resize(bi, (1280, 720))
    ois = cv2.resize(oi, (1280, 720))
    cv2.imshow('bi', bis)
    cv2.imshow('OI', ois)


def harris():
    img = cv2.imread(photoPath)
    # detect corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray)
    corners = cv2.cornerHarris(gray1, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    th = corners > 0.01 * corners.max()
    img[th] = [0, 0, 255]
    imgs = cv2.resize(img, (1280, 720))
    cv2.imshow('original', imgs)


# Functions Block End

imageLabel = Label(image=img, height=400)
imageLabel.grid(row=0, column=0)

# loadButton = Button(root, text='Load', padx=25, command=loadImage)
loadGButton = Button(root1, text='Load Grey', padx=20, command=loadGreyImage)
loadRGBButton = Button(root1, text='Load RGB', padx=20, command=loadRGBImage)
loadHSVButton = Button(root1, text='Load HSV', padx=20, command=loadHSVImage)
loadSplitRGBButton = Button(root1, text='Split RGB Channel', padx=20, command=SplitRGB)
rotateImage = Button(root1, text='Rotate Image', padx=40, command=rotate90)
morphologyx = Button(root1, text='Morphology X', padx=40, command=morphologyx)
morphologicalTransformation = Button(root1, text='morphological Transformation', padx=40,
                                     command=morphologicalTransformation)
sobeldervations = Button(root1, text='sobel dervations', padx=40, command=sobeldervations)
leplacian = Button(root1, text='leplacian', padx=40, command=leplacian)
median = Button(root1, text='median', padx=40, command=median)
inverse = Button(root1, text='inverse', padx=40, command=inverse)
bilateral_filtering = Button(root1, text='bilateral_filtering', padx=40, command=bilateral_filtering)
canny = Button(root1, text='canny', padx=40, command=canny)
dilode = Button(root1, text='dilode', padx=40, command=dilode)
harris = Button(root1, text='harris', padx=40, command=harris)

# firstLabel.grid(row=0,column=0)
# secondLabel.grid(row=3,column=2)


# Showing The Components Block Start

# loadButton.pack()
loadGButton.grid(row=0, column=1)
loadRGBButton.grid(row=0, column=2)
loadHSVButton.grid(row=0, column=3)
loadSplitRGBButton.grid(row=0, column=4)
rotateImage.grid(row=1, column=1)
morphologyx.grid(row=1, column=2)
morphologicalTransformation.grid(row=1, column=3)
sobeldervations.grid(row=1, column=4)
leplacian.grid(row=2, column=1)
median.grid(row=2, column=2)
inverse.grid(row=2, column=3)
bilateral_filtering.grid(row=2, column=4)
canny.grid(row=3, column=1)
dilode.grid(row=3, column=2)
harris.grid(row=3, column=3)

# Showing The Components Block End

root1.mainloop()

# R.I.P Abdelrhman Hafez
