import cv2
import numpy as np

def rotate_image(image, angle=15):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)
    return flipped_image

def preprocess_image(image, apply_rotation=True, apply_flip=True, rotation_angle=15):
    """
    Melakukan preprocessing gambar dengan rotasi dan/atau horizontal flip.
    Args:
        image: Gambar input (NumPy array)
        apply_rotation: Boolean, apakah rotasi diterapkan
        apply_flip: Boolean, apakah flip diterapkan
        rotation_angle: Sudut rotasi dalam derajat (default 15)
    Returns:
        Gambar yang telah diproses
    """
    processed_image = image.copy()
    if apply_rotation:
        processed_image = rotate_image(processed_image, rotation_angle)
    if apply_flip:
        processed_image = flip_image(processed_image)
    return processed_image