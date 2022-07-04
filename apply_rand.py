
import randaugment_seg
import numpy as np

RANDOM_POLICY_OPS = (
    'Identity', 'AutoContrast', 'Equalize', 'Rotate',
    'Solarize', 'Color', 'Contrast', 'Brightness',
    'Sharpness', 'ShearX', 'TranslateX', 'TranslateY',
    'Posterize', 'ShearY'
)


def apply_randomaugment_seg(data, cutoutsz=32, cutoutno=3):
  # input should be a single example in the form  
  # images = [image, label1, label2, ...]
  # image is needs to be normalized to [-1, 1]
  # data example: [images, [poli]] where [(img/255.0-0.5)*2, lab, lab*0+1]

  images, epoch_policy = data
  final_img = randaugment_seg.apply_policy(epoch_policy, images)
  for i in range(cutoutno):
    final_img[0] = randaugment_seg.cutout_numpy(final_img[0], cutoutsz)

  for i in range(1,len(final_img)-1): final_img[i][final_img[-1]<0.5] = 255 # unlabeld part 

  return final_img


def rand_pascal_index(image, magnitude=10, nops=3, cutoutsz=32):
  # image = img # single image 
  # the data is in unit8 (required); output should be in uint8 or uint16
  # has rand_pascal in function name but still applies to cityscapes 
  '''
  example:
  imgnew, indx, indy, indmsk = apply_rand.rand_pascal_index(img)
  labnew = lab[indx, indy]; labnew[indmsk<0.5] = 255;
  '''
    
  policies = [(policy, 0.5, mag) for (policy, mag) in zip( np.random.choice(RANDOM_POLICY_OPS, nops), np.random.randint(1, magnitude, nops))] #
  image = image/255.0*2-1.0 # convert to

  sz = image.shape
  indy, indx = np.meshgrid(np.arange(sz[1]), np.arange(sz[0]))
  indx, indy = indx.astype('uint16'), indy.astype('uint16') # assume size <= 65536
  msk = (indx*0+1).astype('uint8')

  images  = [image, indx, indy, msk]
  final_imgs = apply_randomaugment_seg((images, policies), cutoutsz=cutoutsz)
  final_imgs[0] = ((final_imgs[0] * 0.5 + 0.5) * 255.0).astype('uint8') # convert back to unit8

  return final_imgs





