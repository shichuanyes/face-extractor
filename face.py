import argparse
import glob
import os

import cv2
import imutils
from tqdm import tqdm


def extract_faces(face_detector, image, resolution):
    face_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=2, minNeighbors=5, minSize=(256, 256),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
    for face in faces:
        face_padded = pad_face(image, face, (resolution, resolution))
        face_list.append(face_padded)
    return face_list


def pad_face(image, face, shape):
    x, y, w, h = face
    dy, dx = shape
    Y, X, _ = image.shape
    if w < dx:
        x_mu = x + w // 2
        x = 0 if x_mu - dx // 2 < 0 else X - dx if x_mu + dx // 2 > X else x_mu - dx // 2
    if h < dy:
        y_mu = y + h // 2
        y = 0 if y_mu - dy // 2 < 0 else Y - dy if y_mu + dy // 2 > Y else y_mu - dy // 2
    return image[y:y + max(dy, h), x:x + max(dx, w)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument(
        'input_dir',
        type=str,
        help="input directory containing the input files"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="output directory containing the output files"
    )
    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        default=6,
        help="sample rate for frame sampling"
    )
    parser.add_argument(
        '-n', '--num-best',
        type=int,
        default=20,
        help="max number of output images sorted using Laplacian matrix"
    )
    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=512,
        help="resolution of the output image"
    )
    args = parser.parse_args()

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.wmv')
    for i, file in enumerate(glob.glob(os.path.join(args.output_dir, "*"))):
        _, ext = os.path.splitext(file)
        if ext.lower() not in VIDEO_EXTENSIONS:
            continue

        print(f"Extracting from video {file}...")
        cap = cv2.VideoCapture(file)

        face_list = []
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.sample_rate) as pbar:
            counter = 0
            while cap.grab():
                counter += 1
                if counter % args.sample_rate != 0:
                    continue
                _, img = cap.retrieve()
                face_list += extract_faces(face_detector, img, args.resolution)
                pbar.update(1)
        cap.release()

        print(f"Selecting best {args.num_best} of {len(face_list)} faces for {file}...")
        clarity_scores = []
        for face in tqdm(face_list):
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_scores.append(clarity)

        sorted_indices = sorted(range(len(clarity_scores)), key=lambda k: clarity_scores[k], reverse=True)
        top_faces = [face_list[i] for i in sorted_indices[:args.num_best]]

        for j, face in enumerate(top_faces):
            face = imutils.resize(face, width=args.resolution)
            cv2.imwrite(os.path.join(args.output_dir, f"{i}_{j}.jpg"), face)
