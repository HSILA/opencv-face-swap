import dlib
import cv2
import json

def extract_landmarks(image, output_path='landmarks.json'):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    try:
        # We suppose that there is only one face in the image
        face = detector(image_gray)[0]
        landmarks = predictor(image_gray, face)
    except:
        return "Error"
    
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    output = {
        'point': landmarks_points
    }
    with open(output_path, 'w') as f:
        json.dump(output, f)

    return "Sucess"

if __name__ == "__main__":
    image1 = cv2.imread('images/obama.png')
    extract_landmarks(image1, 'images/obama_auto.json')
    image2 = cv2.imread('images/clinton.png')
    extract_landmarks(image2, 'images/clinton_auto.json')