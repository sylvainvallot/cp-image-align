import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from skimage import io
from skimage.measure import ransac
from skimage.transform import warp, AffineTransform

class Image():
    def __init__(self, image_path) -> None:
        self.content = io.imread(image_path)
        self.name = image_path.split('/')[-1]

class WarpedImage():
    def __init__(self) -> None:
        self.raw = None
        self.cropped = None
        self.name = None

class SetAlignmentPoints():
    def __init__(self, image_path, points_file=None):
        self.filename = points_file
        self.file = self.load_file()
        self.points = []
        self.image = Image(image_path)
        self.get_points()
    
    def load_file(self):
        tree = ET.parse(self.filename)
        return tree.getroot()
            
    def get_points(self):
        for child in self.file:
            self.points.append([float(child.attrib.get('x')), float(child.attrib.get('y'))])
        self.points = np.array(self.points)
        return self.points

class AlignImages():
    """Align images using affine transformation
    """
    def __init__(self, background, layer, ransac=True, debug=False):
        self.background = background
        self.layer = layer
        self.transform_matrix = None
        self.warped = WarpedImage()
        self.ransac = ransac
        self.debug = debug
        self.model()
        self.path = './output/'
        self.plt = None

    def model(self):
        """Compute the transformation matrix between the background and the layer images
        """
        if self.ransac == False:
            self.transform_matrix = AffineTransform()
            self.transform_matrix.estimate(self.background.points, self.layer.points)
            if self.debug:
                print("Affine transform:")
                print(f'Scale: ({self.transform_matrix.scale[0]:.4f}, {self.transform_matrix.scale[1]:.4f}), '
                    f'Translation: ({self.transform_matrix.translation[0]:.4f}, '
                    f'{self.transform_matrix.translation[1]:.4f}), '
                    f'Rotation: {self.transform_matrix.rotation:.4f}')
            return self.transform_matrix
        
        if self.ransac:
            self.transform_matrix, inliers = ransac((self.background.points, self.layer.points), AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
            if self.debug:
                print("RANSAC:")
                print(f'Scale: ({self.transform_matrix.scale[0]:.4f}, {self.transform_matrix.scale[1]:.4f}), '
                    f'Translation: ({self.transform_matrix.translation[0]:.4f}, '
                    f'{self.transform_matrix.translation[1]:.4f}), '
                    f'Rotation: {self.transform_matrix.rotation:.4f}')
        return self.transform_matrix

    def align(self, output_shape=None) -> np.ndarray:
        """_summary_

        Args:
            output_shape (_type_, optional): Desired output shape. Defaults to None.

        Returns:
            np.ndarray: Warped Image
        """
        self.warped.raw = warp(self.background.image.content, inverse_map=self.transform_matrix.inverse, output_shape=output_shape, preserve_range=True)
        self.warped.name = self.background.image.name.split('.')[0] + '_warped.tif'
        return self.warped
    
    def plot(self, alpha=0.9, stacked=True, colorMap='gray'):
        """Plot results of image alignment

        Args:
            alpha (float, optional): Àlpha value of the layer image (Between 0 and 1). Defaults to 0.9.
            stacked (bool, optional): Display only the stacked image. Defaults to True.
        """
        if stacked == False:
            self.warped.cropped = self.warped.raw[0:self.layer.image.content.shape[0], 0:self.layer.image.content.shape[1]]
            fig, ax = plt.subplots(nrows=1, ncols=3)
            
            def plot_ax(ax, image, title):
                for _ in image:
                    ax.imshow(_, cmap=colorMap)
                ax.set_title(title)
                ax.axis('off')
                return
            
            ax[0].imshow(self.warped.raw, cmap=colorMap)
            ax[0].imshow(self.layer.image.content, alpha=alpha, cmap=colorMap)
            ax[0].set_title('Stacked Image')
            ax[0].axis('off')
            
            plot_ax(ax[1], [self.warped.cropped], 'Background Image')
            plot_ax(ax[2], [self.layer.image.content], 'Layer Image')
            
            plt.show()
            self.plt = fig
            return self
        
        else:
            fig, ax = plt.subplots()
            ax.imshow(self.warped.raw)
            ax.imshow(self.layer.image.content, alpha=alpha)
            plt.show()
            self.plt = fig
            return self
        
    def save_plot(self):
        """Save the plot image
        """
        #TODO: output path create folder if not exist
        method_used = 'base'
        if self.ransac == True:
            method_used =  'ransac'
        Path('./output').mkdir(parents=True, exist_ok=True)
        return self.plt.savefig(f'./output/plot_{method_used}.png', bbox_inches='tight')
    
    def export(self, folder_name):
        save_path = self.path + folder_name
        Path(save_path).mkdir(parents=True, exist_ok=True)
        print(f'Saving images to {save_path} with name {self.warped.name} and size {self.warped.cropped.shape}')
        io.imsave(save_path + f'/{self.warped.name}', self.warped.cropped)
        # for _ in [self.background.image, self.layer.image]:
        #     filename = _.name.split('.')[0] + '_aligned.tif'
        #     io.imsave(self.path + '/' + filename, _.content)
        