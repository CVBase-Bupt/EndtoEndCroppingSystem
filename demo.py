import warnings

with warnings.catch_warnings():
    import os, sys
    import cv2
    import tensorflow as tf
    import keras
    from utils import *
    import numpy as np
    from models import config
    from models import model as M
    from PIL import Image, ImageDraw
    from keras.models import Model
    from keras.preprocessing.image import array_to_img

C = config.Config()

def get_shape(img, w, h, scale, ratio):

    size = (w, h)
    if ratio:
        if scale is not None:
            size = (scale, scale)
    else:
        if scale is not None:
            if w <= h:
                size = (scale, scale * h // w)
            else:
                size = (scale * w // h, scale)
    return img.resize(size, Image.ANTIALIAS)


def runn(model, images):
    if C.log:
        result = []
    if os.path.isdir(images):
        test_db = os.listdir(images)
        test_db = [os.path.join(images, i) for i in test_db]
    elif os.path.isfile(images) and images.endswith(C.pic_extend):
        test_db = list(images)
    else:
        raise Exception('Image file or directory not exist.')
    for image_name in test_db:

        if not image_name.endswith(C.pic_extend):
            continue
        img_object = Image.open(image_name)
        w3, h3 = img_object.size
        img_object = img_object.convert('RGB')
        #imgs = img.copy()
        img_reshape = get_shape(img_object, w3, h3, C.scale, C.ratio)
        image = np.asarray(img_reshape)

        h1, w1 = image.shape[0], image.shape[1]
        

        h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16
        image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        boxes = model.predict(image, batch_size=1, verbose=0)

        offset = boxes[0][0]
        saliency_box = boxes[1][0]
        saliency_box = saliency_box * 16.0
        saliency_box[2] = saliency_box[0] + saliency_box[2]
        saliency_box[3] = saliency_box[1] + saliency_box[3]

        saliency_box = [int(y) for y in saliency_box]
        saliency_box = [saliency_box[0], saliency_box[2], saliency_box[1], saliency_box[3]]#x1, x2, y1, y2

        offset = np.array(offset)
        saliency_box = normalization(w2 - 1, h2 - 1, saliency_box)
        aes_bbox = add_offset(w2 - 1, h2 - 1, saliency_box, offset)
        
        img_name = image_name.split('/')[-1]
        if C.log:
            to_file = ' '.join([img_name] + [str(u) for u in saliency_box] + [str(y) for y in aes_bbox])
            result.append(to_file)
        if C.draw:
            if not os.path.isdir(C.box_out_path):
                os.makedirs(C.box_out_path)
            aes_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, aes_bbox)  # [int]
            saliency_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, saliency_box)
            img_draw = img_object.copy()
            draw = ImageDraw.Draw(img_draw)
            draw.rectangle(saliency_box, None, C.saliency_box_color)
            draw.rectangle(aes_box, None, C.aesthetics_box_color)
            img_draw.save(os.path.join(C.box_out_path, img_name))
        if C.crop:
            if not os.path.isdir(C.crop_out_path):
                os.makedirs(C.crop_out_path)
            aes_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, aes_bbox)  # [int]
            img_crop = img_object.crop(aes_box)
           #img_crop = img_crop.resize((w3, h3))
            img_crop.save(os.path.join(C.crop_out_path, img_name))
    if C.log:
        if not os.path.isdir(C.log_path):
            os.mkdir(C.log_path)
        with open(C.log_file, 'w') as f:
            f.write('\n'.join(result))


def main(argv=None):

    if len(sys.argv)<=1:
        images = C.image_path
    else:
        images = sys.argv[1]
    model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test').BuildModel()
    model.load_weights(C.model)
    runn(model, images)

if __name__ == "__main__":
    sys.exit(main())
