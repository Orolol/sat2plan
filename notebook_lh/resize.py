from torchvision import transforms

def resize(X,Y,pourcentage=0.5):
    X_resize = transforms.Resize(int(X.size()[1] * pourcentage))(X)
    Y_resize = transforms.Resize(int(Y.size()[1] * pourcentage))(Y)
