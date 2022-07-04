from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image

PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted

RANDOM_POLICY_OPS = (
    'Identity', 'AutoContrast', 'Equalize', 'Rotate',
    'Solarize', 'Color', 'Contrast', 'Brightness',
    'Sharpness', 'ShearX',
    'Posterize', 'ShearY'
)


def random_flip(xs):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return [np.fliplr(x) for x in xs]
  return xs


def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.
  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.
  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  assert False # not implemented 
  padded_img = np.zeros(
      (img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2]))
  padded_img[amount:img.shape[0] + amount,
             amount:img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img



def create_cutout_mask(img_height, img_width, num_channels, size):
  """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.
  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
  #assert img_height == img_width
  if img_height != img_width:
    pass 
    #print('note: img_height is not equal to img_width, whats the error?') 

  # Sample center where cutout mask will be applied
  height_loc = np.random.randint(low=0, high=img_height)
  width_loc = np.random.randint(low=0, high=img_width)

  # Determine upper right and lower left corners of patch
  upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
  lower_coord = (min(img_height, height_loc + size // 2),
                 min(img_width, width_loc + size // 2))
  mask_height = lower_coord[0] - upper_coord[0]
  mask_width = lower_coord[1] - upper_coord[1]
  assert mask_height > 0
  assert mask_width > 0

  mask = np.ones((img_height, img_width, num_channels))
  mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = 0
  return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
  """Apply cutout with mask of shape `size` x `size` to `img`.
  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.
  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be
  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
  if size <= 0:
    return img
  assert len(img.shape) == 3
  img_height, img_width, num_channels = img.shape
  mask = create_cutout_mask(img_height, img_width, num_channels, size)[0]
  return img * mask


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)


def pil_wrap(imgs, mean=0.5, std=0.5):
  """Convert the `img` numpy tensor to a PIL Image."""
  return [Image.fromarray(np.uint8((imgs[0] * std + mean) * 255.0)).convert('RGBA')] + [Image.fromarray(img) for img in imgs[1:]]


def pil_unwrap(pil_imgs, mean=0.5, std=0.5):
  """Converts the PIL img to a numpy array."""
  s = pil_imgs[0].size # the order has changed
  pic_array = (np.array(pil_imgs[0].getdata()).reshape((s[1], s[0], 4)) / 255.0) 
  i1, i2 = np.where(pic_array[:, :, 3] == 0)
  pic_array = (pic_array[:, :, :3] - mean) / std
  pic_array[i1, i2] = [0, 0, 0]
  labs = [np.array(pil_img.getdata()).reshape((s[1], s[0])) for pil_img in pil_imgs[1:]]
  return [pic_array] + labs


def apply_policy(policy, imgs):
  """Apply the `policy` to the numpy `img`.
  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what
      strength the operation to apply.
    img: Numpy image that will have `policy` applied to it.
  Returns:
    The result of applying `policy` to `img`.
  """
  pil_imgs = pil_wrap(imgs)
  for xform in policy:
    assert len(xform) == 3
    name, probability, level = xform
    xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level)
    pil_imgs = xform_fn(pil_imgs)
  return pil_unwrap(pil_imgs)


class TransformFunction(object):
  """Wraps the Transform function for pretty printing options."""

  def __init__(self, func, name):
    self.f = func
    self.name = name

  def __repr__(self):
    return '<' + self.name + '>'

  def __call__(self, pil_imgs):
    return self.f(pil_imgs)


class TransformT(object):
  """Each instance of this class represents a specific transform."""

  def __init__(self, name, xform_fn):
    self.name = name
    self.xform = xform_fn

  def pil_transformer(self, probability, level):

    def return_function(im):
      if random.random() < probability:
        im = self.xform(im, level)
      return im

    name = self.name + '({:.1f},{})'.format(probability, level)
    return TransformFunction(return_function, name)

  def do_transform(self, images, level):
    f = self.pil_transformer(PARAMETER_MAX, level)
    return pil_unwrap(f(pil_wrap(images)))


################## Transform Functions ##################
identity = TransformT('Identity', lambda pil_imgs, level: pil_imgs)
flip_lr = TransformT(
    'FlipLR', lambda pil_imgs, level: [pil_img.transpose(Image.FLIP_LEFT_RIGHT) for pil_img in pil_imgs])
flip_ud = TransformT(
    'FlipUD', lambda pil_imgs, level: [pil_img.transpose(Image.FLIP_TOP_BOTTOM) for pil_img in pil_imgs])
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast', lambda pil_imgs, level: [ImageOps.autocontrast(pil_imgs[0].convert('RGB')).convert('RGBA')]+pil_imgs[1:] )
equalize = TransformT(
    'Equalize', lambda pil_imgs, level: [ImageOps.equalize( pil_imgs[0].convert('RGB')).convert('RGBA')]+pil_imgs[1:] )
invert = TransformT(
    'Invert', lambda pil_imgs, level: [ImageOps.invert(pil_imgs[0].convert('RGB')).convert('RGBA')]+pil_imgs[1:] )
# pylint:enable=g-long-lambda
blur = TransformT('Blur', lambda pil_imgs, level: [pil_imgs[0].filter(ImageFilter.BLUR)]+pil_imgs[1:])
smooth = TransformT('Smooth', lambda pil_imgs, level: [pil_imgs[0].filter(ImageFilter.SMOOTH)]+pil_imgs[1:] )


def _rotate_impl(pil_imgs, level):
  """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
  degrees = int_parameter(level, 30)
  if random.random() > 0.5:
    degrees = -degrees
  return [pil_img.rotate(degrees) for pil_img in pil_imgs]


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_imgs, level):
  """Applies PIL Posterize to `pil_img`."""
  level = int_parameter(level, 4)
  return [ImageOps.posterize(pil_imgs[0].convert('RGB'), 4 - level).convert('RGBA')] + pil_imgs[1:]


posterize = TransformT('Posterize', _posterize_impl)


def _shear_x_impl(pil_imgs, level):
  """Applies PIL ShearX to `pil_img`.
  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from [0,
      `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return [pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0)) for pil_img in pil_imgs]


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_imgs, level):
  """Applies PIL ShearY to `pil_img`.
  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from [0,
      `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return [pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0)) for pil_img in pil_imgs]



shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_imgs, level):
  """Applies PIL TranslateX to `pil_img`.
  Translate the image in the horizontal direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from [0,
      `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return [pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0)) for pil_img in pil_imgs]


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_imgs, level):
  """Applies PIL TranslateY to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from [0,
      `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return [pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level)) for pil_img in pil_imgs]


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_imgs, level, interpolation=Image.BILINEAR):
  """Applies a crop to `pil_img` with the size depending on the `level`."""
  cropped = [pil_img.crop((level, level, pil_img.size[0] - level, pil_img.size[1] - level)) for pil_img in pil_imgs]
  resized = [cropped[0].resize(pil_imgs[0].size, interpolation)] + [pil_img.resize(pil_img.size, Image.NEAREST) for pil_img in pil_imgs[1:]] # for the label: Image.NEAREST 
  return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_imgs, level):
  """Applies PIL Solarize to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from [0,
      `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had Solarize applied to it.
  """
  level = int_parameter(level, 256)
  return [ImageOps.solarize(pil_imgs[0].convert('RGB'), 256 - level).convert('RGBA')] + pil_imgs[1:]


solarize = TransformT('Solarize', _solarize_impl)


def _cutout_pil_impl(pil_imgs, level):
  """Apply cutout to pil_img at the specified level."""
  size = int_parameter(level, 20)
  if size <= 0:
    return pil_imgs
  pil_img = pil_imgs[0]
  img_width, img_height = pil_img.size
  num_channels = 3
  _, upper_coord, lower_coord = (
      create_cutout_mask(img_height, img_width, num_channels, size))
  pixels = pil_img.load()  # create the pixel map
  for i in range(upper_coord[0], lower_coord[0]):  # for every column
    for j in range(upper_coord[1], lower_coord[1]):  # For every row
      pixels[i, j] = (127, 127, 127, 0)  # set the colour accordingly
  return [pil_img] + pil_imgs[1:]


cutout = TransformT('Cutout', _cutout_pil_impl)


def _enhancer_impl(enhancer):
  """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

  def impl(pil_imgs, level):
    v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
    return [enhancer(pil_imgs[0]).enhance(v)] + pil_imgs[1:]

  return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
    identity, flip_lr, flip_ud, auto_contrast, equalize, invert, rotate,
    posterize, crop_bilinear, solarize, color, contrast, brightness, sharpness,
    shear_x, shear_y, translate_x, translate_y, cutout, blur, smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()



