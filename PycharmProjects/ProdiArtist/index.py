from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import IPython.display as display
import numpy as np
import PIL.Image

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


def get_output_from_layer_name(layer_name):
    return vgg.get_layer(layer_name).output


def load(path):
    max_dim = 1024
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def extract(img):
    content_outputs = get_content_outputs(img)
    style_outputs = get_style_outputs(img)
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    content_dict = {content_name: value for content_name, value in zip(content_layers, content_outputs)}

    style_dict = {style_name: value for style_name, value in zip(style_layers, style_outputs)}
    return {"content": content_dict, "style": style_dict}


def get_content_outputs(img):
    content_outputs = [get_output_from_layer_name(layer_name) for layer_name in content_layers]
    vgg_content = tf.keras.Model([vgg.input], content_outputs)
    vgg_content.trainable = False
    return vgg_content(tf.keras.applications.vgg19.preprocess_input(img * 255.0))


def get_style_outputs(img):
    style_outputs = [get_output_from_layer_name(layer_name) for layer_name in style_layers]
    vgg_style = tf.keras.Model([vgg.input], style_outputs)
    vgg_style.trainable = False
    return vgg_style(tf.keras.applications.vgg19.preprocess_input(img * 255.0))


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def generate(img):
    return tf.Variable(img);


def getloss(real, generated):
    content_loss = tf.add_n(
        [tf.reduce_mean((real["content"][layer_name] - generated["content"][layer_name]) ** 2) for layer_name in generated["content"].keys()])
    style_loss = tf.add_n(
        [tf.reduce_mean((real["style"][layer_name] - generated["style"][layer_name]) ** 2) for layer_name in generated["style"].keys()])
    return (content_loss * 1e4) + ((style_loss * 1e-2) / 5)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def train(img):
    with tf.GradientTape() as tape:
        generated_img = extract(img)
        loss = getloss({"content": real_content, "style": real_style}, generated_img)
        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))


def show(img):
    img = img * 255
    img = np.array(img, dtype=np.uint8)
    if np.ndim(img) > 3:
        assert img.shape[0] == 1
        img = img[0]
    return PIL.Image.fromarray(img)


vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
content_image = load(tf.keras.utils.get_file('thomas-west-virginia-SMALLTOWNUSA0517.jpg',
                                             'https://cdn-image.travelandleisure.com/sites/default/files/styles/1600x1000/public/1494534190/thomas-west-virginia-SMALLTOWNUSA0517.jpg'))
style_image = load(tf.keras.utils.get_file('48e7f9a80e43867265caeb4cc6250f7c.jpg',
                                           'https://i.pinimg.com/originals/48/e7/f9/48e7f9a80e43867265caeb4cc6250f7c.jpg'))
generated = generate(content_image)
real_content = extract(content_image)["content"]
real_style = extract(style_image)["style"]
for i in range(100):
    train(generated)
show(generated).save('stylized-image.png')
