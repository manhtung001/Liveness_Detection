import face_recognition
from os import listdir
from os.path import isfile, join
from glob import glob

def get_users():

    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    print("[LOG] Encoding faces ...")

    for i in glob("people/*.jpg"):
        # Load image
        image = cv2.imread(i)
        # Convert it from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect face in the image and get its location (square boxes coordinates)
        boxes = face_recognition.face_locations(image, model='cnn')

        # Encode the face into a 128-d embeddings vector
        encoding = face_recognition.face_encodings(image, boxes)

        if len(encoding) > 0:
            known_encodings.append(encoding[0])
            known_names.append(i[7:-4])

        encodings = {"encodings": known_encodings, "names": known_names}
        np.save('encodings.npy', encodings)

    return encodings


        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]

