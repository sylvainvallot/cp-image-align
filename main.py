import cv2
from sliping import CreateImage, SetAlignmentPoints, align_images

def main(background, layer):
    image1 = CreateImage(background, 'img1')
    image2 = CreateImage(layer, 'img2')

    ebsd = SetAlignmentPoints(image1.path, image1.name)
    slipband = SetAlignmentPoints(image2.path, image2.name)

    ebsd.showWindow()
    slipband.showWindow()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(ebsd.points)
    print(slipband.points)

    aligned_result, transform_matrix = align_images(ebsd.image, slipband.image, ebsd.points, slipband.points)
    print('TransformMatrix: ', transform_matrix)

    cv2.imshow('Aligned Image', aligned_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
if __name__ == "__main__":
    
    BACKGROUND_EBSD = "data/img1.tif"
    LAYER_SLIPBANDS = "data/img2.tif"

    main(BACKGROUND_EBSD, LAYER_SLIPBANDS)
