from PIL import Image
import torchvision.transforms as transforms

def load_image(path, size=(32,32)):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # batch dimension
