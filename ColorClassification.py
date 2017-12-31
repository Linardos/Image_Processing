"""
The outline of the code follows structured coding principles. All functions are called within main.
"""

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Object Classification based on Color')
parser.add_argument('image', metavar='image',
                    help='path to image')
parser.add_argument('-C', metavar='color_margin', default=[0,0,0], type=int, nargs=3,
                    help='Input 3 numbers, one for each RGB color. These will be used for thresholding color, Green Blue and Red. By increasing each value you keep pixels with higher intensity to the respective color, default = [0, 0, 0]')
parser.add_argument('-k1', metavar='Open', type=int, default=[0,0,0], nargs=3, help='Specifies the size of kernel 1 for each color, used for denoising (opening), (default: [0, 0, 0])')
parser.add_argument('-k2', metavar='Close', type=int, default=[0,0,0], nargs=3,help='Specifies the size of kernel 2 for each color, used for denoising (closing), (default: [0, 0, 0])')
parser.add_argument('-cells', dest='cells', action='store_true',
                    help='Set to an alternate mode for recognizing distinct cells and their nuclei (default: False)')

def main():
    global args
    args = parser.parse_args()

    image = cv2.imread(args.image) #load the image

    gkernels = (args.k1[0], args.k2[0])
    bkernels = (args.k1[1], args.k2[1])
    rkernels = (args.k1[2], args.k2[2])

    CurrentImage = ImageParameters(image, args.C, [gkernels,bkernels,rkernels])

    img_b, img_g, img_r = cv2.split(CurrentImage.image) #bgr protocol

    ShowImage(np.hstack([img_b, img_g, img_r]))

    if args.cells:
        #This is a special case for the image with the cells. There was a lot of manual handling I had to do in order to achieve satisfactory results so this is an exception and no parameters are passed by the user.
        #The blue image includes are the nuclei so we separate the two
        #I zoomed in the pixels to figure out the best value range to separate the 2 nuclei
        RedNucleus = img_b.copy()
        RedNucleus[img_r<80] = 0
        RedNucleus[img_b<100] = 0
        RedNucleus[img_g>100] = 0
        BlueNucleus = img_b.copy()
        BlueNucleus[img_r>100] = 0
        BlueNucleus[img_b<100] = 0
        BlueNucleus[img_g<20] = 0
        ShowImage(np.hstack([RedNucleus, BlueNucleus]))

        #Now to binarize, all values that arent 0 will be 250:
        RedNucleus[RedNucleus!=0]=250
        BlueNucleus[BlueNucleus!=0]=250
        ShowImage(np.hstack([RedNucleus, BlueNucleus]))

    ShowImage(np.hstack([img_b, img_g, img_r]))

    BlueMask = np.zeros(np.shape(img_r),np.uint8) #creating mask array
    test1=img_r/2+img_g/2 + CurrentImage.margins[0]
    #keep the pixels were blue overpowers the other 2 colors
    BlueMask[img_b > test1] = 250
    #and also the pixels were blue is simply very high

    GreenMask = np.zeros(np.shape(img_r),np.uint8)
    test1=img_b/2+img_r/2 + CurrentImage.margins[1]
    GreenMask[img_g>test1] = 250

    RedMask = np.zeros(np.shape(img_r),np.uint8)
    test1=img_b/2+img_g/2 + CurrentImage.margins[2]
    RedMask[img_r>test1] = 250

    masks = [BlueMask, GreenMask, RedMask]
    ShowImage(np.hstack(masks))

    output = []
    contoured = []

    #Calculate the total area in order to calculate the surface coverage later
    blank_image = np.ones(np.shape(BlueMask), np.uint8)*255
    _, blank_contour, _ = cv2.findContours(blank_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    total_area = cv2.contourArea(blank_contour[0])
    print("total area is %i  "%total_area)

    if args.cells:
        #This block of code is for when the user defined the mode of cell/nuclei recognition
        GreenCell = cv2.bitwise_or(GreenMask, BlueNucleus)
        RedCell = cv2.bitwise_or(RedMask, RedNucleus)
        ShowImage(np.hstack([GreenCell, RedCell]))

        Cells = [GreenCell, RedCell, BlueNucleus, RedNucleus]

        object_areas = [] #We want to analyze the coverage of nuclei with respect to cells, so we need to handle this differently.
        for i, cell in enumerate(Cells):
            if i<2:
                cell = CloseHoles(cell, args.k1[i], args.k2[i])
            else:
                cell = CloseHoles(cell, args.k1[2], args.k2[2])

            output.append(cv2.bitwise_and(CurrentImage.image, CurrentImage.image, mask = cell)) #bitwise_and applies the mask to the image
            image, contours, hierarchy = cv2.findContours(cell,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Find the contours to outline the objects

            todraw = CurrentImage.image.copy() #make a copy of the original image
            cv2.drawContours(todraw, contours, -1, (0,255,0), 2)
            contoured.append(todraw)

            object_area = 0
            for i in contours:
                object_area += cv2.contourArea(i) #Calculate the area this object covers
            object_areas.append(object_area)

        #calculate the coverage percentage
        coverage_percentage = round((object_areas[2] / object_areas[0])*100, 2)
        print('The blue nuclei cover {} percent of the total green cell area'.format(coverage_percentage))
        coverage_percentage = round((object_areas[3] / object_areas[1])*100, 2)
        print('The purple nuclei cover {} percent of the total red cell area'.format(coverage_percentage))

    else:
        for i, mask in enumerate(masks):
            k1,k2 = CurrentImage.openclose[i]
            mask = CloseHoles(mask, k1,k2)

            output.append(cv2.bitwise_and(CurrentImage.image, CurrentImage.image, mask = mask)) #bitwise_and applies the mask to the image
            image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Find the contours to outline the objects

            todraw = CurrentImage.image.copy() #make a copy of the original image
            cv2.drawContours(todraw, contours, -1, (0,255,0), 2)
            contoured.append(todraw)

            object_area = 0
            for i in contours:
                object_area += cv2.contourArea(i) #Calculate the area this object covers
            print(object_area)

            #calculate the coverage percentage
            coverage_percentage = round((object_area / total_area)*100, 2)

            print("{}% Coverage".format(coverage_percentage))

    ShowImage(np.hstack(output))
    ShowImage(np.hstack(contoured))

class ImageParameters:
    #A class to include all the parameters given by the user. Not a necessary structure but helps me organize stuff.
    def __init__(self, image, margins, openclose):
        self.image = image
        self.margins = margins
        self.openclose = openclose

def ShowImage(image):
    #A function for the standard image visualization
    cv2.namedWindow( "Image Display", cv2.WINDOW_NORMAL )
    cv2.imshow( "Image Display", image )
    cv2.waitKey()

def LocateContours(mask, i):
    #Find the contours based on the mask that was created
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    todraw = CurrentImage.image.copy() #make a copy of the original image
    cv2.drawContours(todraw, contours, -1, (0,255,0), 1)
    ShowImage(todraw)
    #Calculate total area the contours encircle
    object_area = 0
    for i in contours:
        object_area += cv2.contourArea(i)
    print(object_area)
    return(todraw, object_area)


def CloseHoles(mask, k1=5, k2=3):
    #Close small holes
    """
    If the user specified a valid kernel size proceed with the denoising process
    """
    starting_image = mask

    if k1 > 0:
        kernel_1 = np.ones((k1,k1),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_1, iterations = 1)
        ShowImage(np.hstack([starting_image, mask]))
    if k2 > 0:
        kernel_2 = np.ones((k2,k2),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_2, iterations = 1)
        ShowImage(np.hstack([starting_image, mask]))

    return(mask)


if __name__ == '__main__':
    main()
