import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_generator import ModelGenerator

def catalog_image(img_path):

    model = ModelGenerator()
    model.load_state_dict(torch.load('./models/77-accuracy.pt', map_location=lambda storage, loc: storage), strict=False)
    
    emotion_dict = {0: 'ANGER', 1: 'DISGUEST', 2: 'FEAR', 3: 'HAPPY',
                    4: 'SAD', 5: 'SURPRISE', 6: 'NEUTRAL'}

    val_transform = transforms.Compose([
        transforms.ToTensor()])


    img = cv2.imread(img_path)

    print(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
        resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        X = resize_frame/256
        X = Image.fromarray((resize_frame))
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())]
            print(pred)
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 232, 255), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    catalog_image('images/surprise.jpg')