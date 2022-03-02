import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def transfer_style(content_image, style_image, model_path):

    """
    :param content_image: content image as numpy array
    :param style_image: style image as numpy array
    :param model_path: path to the downloaded pre-trained model.

    The 'model' directory already contains the downloaded pre-trained model,but 
    you can also download the pre-trained model from the below TF HUB link:
    https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2

    :return: A Styled image as 3D numpy array.

    """

    #--------------------------------------------------------------

    # resize the images to (1000,1000) if greater than (2000 x 2000)

    size_threshold = 2000
    resizing_shape = (1000,1000)
    content_shape = content_image.shape
    style_shape = style_image.shape

    resize_content = True if content_shape[0] > size_threshold or content_shape[1] > size_threshold else False
    resize_style = True if style_shape[0] > size_threshold or style_shape[1] > size_threshold else False

    if resize_content is True:
        print("Content Image bigger than (2000x2000), resizing to (1000x1000)")
        content_image = cv2.resize(content_image,(resizing_shape[0],resizing_shape[1]))
        content_image = np.array(content_image)
    
    if resize_style is True :
        print("Style Image bigger than (2000x2000), resizing to (1000x1000)")
        style_image = cv2.resize(style_image,(resizing_shape[0],resizing_shape[1]))
        style_image = np.array(style_image)

    #--------------------------------------------------------------


    print("Resizing and Normalizing images...")
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    style_image = tf.image.resize(style_image, (256, 256))

    print("Loading pre-trained model...")
    # The hub.load() loads any TF Hub model
    hub_module = hub.load(model_path)

    print("Generating stylized image now...wait a minute")
    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # reshape the stylized image
    stylized_image = np.array(stylized_image)
    stylized_image = stylized_image.reshape(
        stylized_image.shape[1], stylized_image.shape[2], stylized_image.shape[3])

    print("Stylizing completed...")
    return stylized_image

