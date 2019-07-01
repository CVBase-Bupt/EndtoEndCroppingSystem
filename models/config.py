import os
class Config:

    def __init__(self):

        self.scale = 224
        self.ratio = False
        self.gamma = 3.0
        self.theta = 0.01

        self.model = self.set_model()

        self.draw = False
        self.crop = False
        self.log = True

        self.pic_extend = ('jpg', 'png', 'jpeg', 'bmp')

        self.image_path = 'images'
        self.log_path = 'logs'
        self.log_file = os.path.join(self.log_path, 'log.txt')
        self.box_out_path = 'box_results'
        self.crop_out_path = 'crop_results'

        self.saliency_box_color = 'blue'
        self.aesthetics_box_color = 'yellow'

    def set_model(self):
        path_base = 'weights/model_'
        path_ext = '.h5'
        path_scale = str(self.scale)
        path_ratio = 'square' if self.ratio else ''
        return path_base + path_ratio + '_' + path_scale + path_ext
