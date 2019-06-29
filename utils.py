import numpy as np


def add_offset(w, h, bbox, offset):

    crop_h = int(h * (float(bbox[1]) - float(bbox[0])))
    crop_w = int(w * (float(bbox[3]) - float(bbox[2])))

    new_w = crop_w / (1 - float(offset[2]) - float(offset[3]) + 1e-10)
    new_h = crop_h / (1 - float(offset[0]) - float(offset[1]) + 1e-10)

    r_w = min(w, max(0, new_w))
    r_h = min(h, max(0, new_h))

    x1 = max(0, h * float(bbox[0]) - r_h * float(offset[0]))
    x2 = min(h, x1 + r_h)
    y1 = max(0, w * float(bbox[2]) - r_w * float(offset[2]))
    y2 = min(w, y1 + r_w)

    bbox_aes = [x1 / float(h), x2 / float(h), y1 / float(w), y2 / float(w)]

    return bbox_aes


def recover_from_normalization_with_order(w, h, bbox):
    box = [max(0, int(bbox[2] * w)), max(0, int(bbox[0] * h)), min(w, int(bbox[3] * w)), min(h, int(bbox[1] * h))]
    return box


def recover_from_normalization(w, h, bbox):
    box = [max(0, int(bbox[0] * h)), min(h, int(bbox[1] * h)), max(0, int(bbox[2] * w)), min(w, int(bbox[3] * w))]
    return box


def normalization(w, h, bbox):
    box = [min(max(0.0, float(bbox[0]) / h), 1.0),
           min(max(0.0, float(bbox[1]) / h), 1.0),
           min(max(0.0, float(bbox[2]) / w), 1.0),
           min(max(0.0, float(bbox[3]) / w), 1.0)]
    return box
