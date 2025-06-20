import yaml

def load_config(path='config/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def preprocess_image(frame, size=(224,224)):
    import cv2
    from torchvision import transforms
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor
