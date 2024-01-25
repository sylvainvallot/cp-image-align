import cv2
import numpy as np

class CreateImage:
    def __init__(self, file, name):
        self.path = file
        self.name = name

class SetAlignmentPoints:
    def __init__(self, file, name, gray=False):
        self.file = file
        self.name = name
        self.points = []
        self.image = cv2.imread(self.file)
        # self.image = cv2.imread(self.file, -1) for gray support

        self.params = {'image': self.image, 'name': name}
        self.gray = gray
        
    def select_points(self, event, x, y, flags, params):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            print(f"Point {len(self.points)} selected at {x}, {y}")
            color = (0, 255, 0)
            markerType = cv2.MARKER_CROSS
            markerSize = 15
            thickness = 2
            cv2.drawMarker(params['image'], (x, y), color, markerType, markerSize, thickness)
            cv2.imshow(params['name'], params['image'])

    def showWindow(self):
        if self.gray:
            scaled_image = ((self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image)) * 255).astype(np.uint8)
            cv2.imshow(self.name, scaled_image)
        else:
            cv2.imshow(self.name, self.image)
        cv2.startWindowThread()
        cv2.setMouseCallback(self.name, self.select_points, self.params)
            

def align_images(background, layer, kps_bg, kps_layer, transparency=0.92):
    # Create a new image to hold the aligned result
    aligned_image = np.copy(background)

    # Calculate transformation matrix using keypoints
    M, _ = cv2.estimateAffinePartial2D(np.array(kps_layer), np.array(kps_bg))

    # Warp the image
    detail_warped = cv2.warpAffine(layer, M, (background.shape[1], background.shape[0]))

    # Blend the images
    cv2.addWeighted(detail_warped, transparency, aligned_image, 1 - transparency, 0, aligned_image)

    return aligned_image, M