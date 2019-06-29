import os
class Config:

    def __init__(self):
        self.model = 'weights/model.h5'
        self.scale = 512
        self.ratio = True
        self.gamma = 3.0
        self.theta = 0.01

        self.draw = True
        self.crop = True
        self.log = True

        self.pic_extend = ('jpg', 'png', 'jpeg', 'bmp')

        self.image_path = 'images'
        self.log_path = 'logs'
        self.log_file = os.path.join(self.log_path, 'log.txt')
        self.box_out_path = 'box_results'
        self.crop_out_path = 'crop_results'

        self.saliency_box_color = 'blue'
        self.aesthetics_box_color = 'yellow'

