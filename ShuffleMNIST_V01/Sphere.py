import torch

class RadiusError(TypeError):
    pass

class Sphere():

    '''
    Sphere class gives the object topological feaures. You can know if points inside tensors
    are in interior, boundary and/or exterior of a ball constructed with supreme norm.

    Parameters
    ---------------

    anchor : the top left corner of the Sphere.
    radius : radius of the sphere.

    Returns
    ---------------

    isboundary : Boolean
        If given anchor is in the sphere.
    isinterior : Boolean
        If given anchor is inside the ball.
    isexterior : Boolean
         If given anchor is outside the ball.
    '''

    def __init__(self, anchor, radius = 0):
        self.anchor = anchor
        self.radius = radius

        if not type(anchor) == torch.Tensor:
            raise TypeError('anchor must be tensors')

        self.corner_x = anchor[0].item()
        self.corner_y = anchor[1].item()
        self.center_x = anchor[0].item() + 14
        self.center_y = anchor[1].item() + 14

        if radius < 0:
            raise ValueError('Invalid radius!')

        if not type(radius) == int:
            raise ValueError('Invalid radius, non integrer value!')


        #if all([type(self.corner_x) == int ,type(self.corner_y) == int]) == False:
        #    raise TypeError('Data in anchor list must be integrers.')


    def center(self):
        return torch.tensor([self.center_x , self.center_y])

    def isinterior(self, centro):
        self.centro = centro
        if max(abs(self.centro[0] - self.center_x), abs(self.centro[1] - self.center_y)) < self.radius:
            return True
        else:
            return False

    def isboundary(self, centro):
        self.centro = centro
        if max(abs(self.centro[0] - self.center_x), abs(self.centro[1] - self.center_y)) == self.radius:
            return True
        else:
            return False

    def isexterior(self, centro):
        self.centro = centro
        if max(abs(self.centro[0] - self.center_x), abs(self.centro[1] - self.center_y)) > self.radius:
            return True
        else:
            return False

        # Add the __str__() method
    def __str__(self):
        return 'Shere with anchor: {}  \n center: {} \n radius: {}'.format([self.corner_x,self.corner_y],[self.center_x,self.center_y],self.radius)

if __name__=='__main__' :
    radius = torch.tensor([10,10])
    sphere = Sphere(radius,30)
    print(sphere)
    print('The point {} is the center an radius {}'.format(sphere.center(), sphere.radius))
