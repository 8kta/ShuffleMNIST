import numpy as np
import os

import torch
from ShuffleMNIST import RandomAnchors

import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from PIL import Image
#

#clase parent cub con tres parámetros, dos dados.
class ShuffleMNIST():

    '''
    This class returns a data set made of images from MNIST, the dataset
    will contain certain number of images given by num parameter pasted
    in a bigger image if size wall_shape x wall_shape. Also the anchors
    can be customised if not want randomness.

    Use
    ---------------
    Your first parameter have to be torch DataLoader from MNIST. When
    ShuffleMNIST has done it can be passed through same DataLoader module
    from pythorch.utils.

    Parameters
    ---------------

    dataloader : dataloader type
            database passed through toch.utils.dataloader

    anchors :  tensor type
            anchors that will be the corners
            in each image, if it is not specified the corners will be
            generated radomly.

    num : int type
            number of images to past. See RandomAnchors class.

    radius : int Type
            radius of sphere. See Sphere class.

    wall_shape : int Type
            size of the output images (wall_shape x wall_shape)

    sum : Boolean
            If true, labels of images will be the sum of numbers in the
            output image. If false, labels will be the product.

    is_train : Boolean
            If true, the class will recognize the train dataloader type.
            If false, it will recognize the test one.

    Returns
    ---------------

    Return a whole data base with __getitem__ and __len__ methods. That is,
    it can be evaluated in torch.utils.dataloader. .
    '''


    def __init__(self, dataloader, anchors, num=1, radius = 0, wall_shape = 112, sum = True, is_train=True):
        self.dataloader = dataloader
        self.num = num
        self.wall_shape = wall_shape
        self.sum = sum
        self.radius = radius
        self.is_train = is_train
        self.anchors = anchors
        self.train_img = []
        self.train_label = []
        self.test_img = []
        self.test_label = []

        #hace la base de datos
        if not type(self.dataloader) == torch.utils.data.dataloader.DataLoader:
            raise TypeError('This need to be torch.utils.data.dataloader.DataLoader')

        if not type(self.wall_shape) == int:
            raise TypeError('Wall_shape must be integrer')

        if not self.wall_shape > 0:
            raise ValueError('Wall_shape is positive')

        if not type(self.sum) == bool:
            raise TypeError('sum parameter is Boolean')

        if not type(self.is_train) == bool:
            raise TypeError('is_train is Boolean')

        if not type(anchors) == list:
            raise TypeError('anchors must be a list')


        images = enumerate(self.dataloader)
        for count, items in images: #937 o 9 veces
            #creamos los indices para agarar 4 imágenes aleatorias
            rand = torch.randint(self.dataloader.batch_size, size=(self.dataloader.batch_size, self.num))
            for r in rand:
                #definimos las imagenes y sus respectivos labels
                lst_im  =  [items[0][r[j]][0] for j in range(self.num)]
                lst_lab =  [items[1][r[j]] for j in range(self.num)]

                if self.sum:
                    label = np.sum(lst_lab)
                    pass
                if not self.sum:
                    label = np.prod(lst_lab)
                    pass

                #Creamos la matriz grande
                wall_da = torch.zeros(self.wall_shape,self.wall_shape)

                #se hacen las esquinas aleatorias y se definen como anclas
                if len(self.anchors) == 0:
                    random = RandomAnchors.RandomAnchors(self.num, self.radius)
                    anchors = random.random_img()
                else:
                    anchors = self.anchors
                    #Aquí tiene que salir un error en el numero de imagenes

                for i in range(self.num):
                    wall_da[anchors[i][0]:anchors[i][0]+lst_im[i].shape[0], anchors[i][1]:anchors[i][1]+lst_im[i].shape[1]] = lst_im[i]

                anchors = self.anchors
                #hace las imagenes y labels
                if self.is_train:
                    self.train_img.append(wall_da)
                    self.train_label.append(label)

                if not self.is_train:
                    self.test_img.append(wall_da)
                    self.test_label.append(label)

    #define un modulo getitem
    def __getitem__(self, index):
        #extrae cada imagen
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]

        else:
            img, target = self.test_img[index], self.test_label[index]

        return img, target #regresa la imagen y el label

    #nuevo modulo dentro de la clase CUP
    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


#if __name__ == '__main__':
#    dataset =  torchvision.datasets.MNIST('Users\8kta\Documents\GitHub\deeplearning_project', train=True, download=False,
#                             transform=torchvision.transforms.ToTensor())
#
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True,drop_last=True)
#    dataset = SuffleMNIST(dataloader, num=4, radius = 42, wall_shape = 112, sum = True)
#
#    for data in dataset:
#        print(data[0].size(), data[1])
#
#    print(len(dataset.train_img))
#    print(len(dataset.train_label))
    #dataset = CUB(root='./CUB_200_2011', is_train=False)
    #print(len(dataset.test_img))
    #print(len(dataset.test_label))
    #for data in dataset:
    #    print(data[0].size(), data[1])
