from fer import FER
import cv2
emotion_detector=FER(mtcnn=True)
test_img=cv2.imread("F://suprise3.jpg") #Here you need to change the image location as per your availability.
analysis=emotion_detector.detect_emotions(test_img)
dominant_emotion,emotion_score=emotion_detector.top_emotion(test_img)
print(dominant_emotion,emotion_score)
