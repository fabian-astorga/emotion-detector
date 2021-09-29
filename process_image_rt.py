from __future__ import print_function
import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model_generator import ModelGenerator

def camera_detection():
    model = ModelGenerator()
    model.load_state_dict(torch.load('./models/mymodel.pt', map_location=lambda storage, loc: storage), strict=False)

    emotion_dict = {0: 'ANGER', 1: 'DISGUEST', 2: 'FEAR', 3: 'HAPPY',
                    4: 'SAD', 5: 'SURPRISE', 6: 'NEUTRAL'}


    val_transform = transforms.Compose([
        transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = emotion_dict[int(top_class.numpy())]
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 232, 255), 3)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    camera_detection()