import tensorflow as tf
import numpy as np
from PIL import Image

from model import ResNetV2
from preprocess_v3 import gen_images

TEST_GIF = 'test-xsag.gif'
WEIGHTS = 'resnet_model2.h5'
ALPHABET = '2345678ABCDEFGHKLMNPQRSTUVWXY'

def prediction(images):
    model = ResNetV2()
    model.build(input_shape=(None,) + (52, 52, 1))
    model.load_weights(WEIGHTS)

    pred = model(images, training=False)
    
    return ''.join(ALPHABET[i] for i in tf.argmax(pred, -1))


def main():
    imgs = gen_images((Image.open(TEST_GIF)))
    print(prediction(imgs))

if __name__ == '__main__':
    main()