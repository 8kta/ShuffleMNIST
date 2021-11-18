import torch
from ShuffleMNIST import Sphere

class RandomAnchors():
    '''
    RandomAnchors class has been made to give random centes for the images
    to be pasted in a tensor wall, this version give entries to the tensor center
    less than 84. It could be generalized, it would happen in a future.

    Parameters
    ---------------

    num : number of images you want in the wall.
    radius : radius of the sphere.

    Returns
    ---------------

    anchors : tensor Type
        These anchors will be the anchos for the images to paste.
    '''
    def __init__(self, num = 1, radius = 42):
        self.num = num
        self.radius = radius
        self.anchors = []
        self.sph_centers = []

        if num < 1 or type(num) != int:
            raise ValueError('Number of images must be an integrer greater or igual from 1.')

    def random_img(self):
        while len(self.anchors) < self.num:
          rand = torch.randint(84, size=(1, 2)).reshape(-1)
          #print(type(rand[0].item()))
          sph_rand = Sphere.Sphere(rand, self.radius)
          center = sph_rand.center()
          sph_center = Sphere.Sphere(center,self.radius)
          if len(self.anchors) == 0:
            self.anchors.append(rand)
            self.sph_centers.append(sph_center)
          else:
            if any(sph_center.isinterior(center) for sph_center in self.sph_centers) == False:
              self.anchors.append(rand)
              self.sph_centers.append(sph_center)
              pass
            else:
              pass
        return self.anchors


if __name__=='__main__' :
    num = 4
    an = RandomAnchors(num)
    random = an.random_img()
    print('{} random anchors are {}'.format(num , random))
