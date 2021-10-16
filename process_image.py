import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import base64
from model_generator import ModelGenerator

def catalog_image(base64_image):
    
    model = ModelGenerator()
    model.load_state_dict(torch.load('./models/77-accuracy.pt', map_location=lambda storage, loc: storage), strict=False)
    
    emotion_dict = {0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Felicidad',
                    4: 'Tristeza', 5: 'Sorpresa', 6: 'Seriedad'}

    val_transform = transforms.Compose([
        transforms.ToTensor()])

    nparr = np.fromstring(base64.b64decode(base64_image), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.imread(base64_image)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 232, 255), 2)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.grid(False)
    # plt.axis('off')
    # plt.show()

    retval, buffer = cv2.imencode('.jpg', img)
    base64_image_result = base64.b64encode(buffer)

    return base64_image_result
