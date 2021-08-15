import cv2
import numpy as np


def snip_bounding_box(image, section, margin=0.3, size=(64, 64)):
    """
    Snip a section of the input image then add margin with specific size
    :param image: input image
    :param section: area in the input image to snip (x top left, y top left, width, height)
    :param margin: margin for the snip section
    :param size: size of output image
    :return: image in numppy array (width x height x 3)
    """
    img_h, img_w, _ = image.shape
    (x, y, w, h) = section
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    margin_w = min(margin_w, x, (img_w - x - w))
    margin_h = min(margin_h, y, (img_h - y - h))
    x = x - margin_w
    y = y - margin_h
    w = w + 2 * margin_w
    h = h + 2 * margin_h
    snipped_section = image[y:y + h, x:x + w]
    resized_section = cv2.resize(snipped_section, size, interpolation=cv2.INTER_AREA)
    resized_section = np.array(resized_section)
    return resized_section, (x, y, w, h)


def create_labeled_bounding_box(image, area_of_bb, label):
    """
    Draw a bounding box with specific label
    :param image: input image
    :param area_of_bb: location of bounding box (x, y , w, h)
    :param label: label of bounding box
    :return: None
    """
    x = area_of_bb[0]
    y = area_of_bb[1]
    w = area_of_bb[2]
    h = area_of_bb[3]
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 190, 0), 2)
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
