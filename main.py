import cv2
import numpy as np

def initialize_video_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {source}")
    return cap

def read_and_resize(img_path, size=None):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    if size:
        img = cv2.resize(img, size)
    return img

def stack_images(img_array, scale, labels=[]):
    sizeW = img_array[0][0].shape[1]
    sizeH = img_array[0][0].shape[0]
    rows, cols = len(img_array), len(img_array[0])
    for r in range(rows):
        for c in range(cols):
            img_array[r][c] = cv2.resize(img_array[r][c], (sizeW, sizeH), None, scale, scale)
            if img_array[r][c].ndim == 2:
                img_array[r][c] = cv2.cvtColor(img_array[r][c], cv2.COLOR_GRAY2BGR)
    hor = [np.hstack(img_array[r]) for r in range(rows)]
    ver = np.vstack(hor)
    if labels:
        each_img_width = ver.shape[1] // cols
        each_img_height = ver.shape[0] // rows
        for r in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c*each_img_width, r*each_img_height),
                              (c*each_img_width + len(labels[r])*13 + 27, 30 + r*each_img_height),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[r], (c*each_img_width + 10, r*each_img_height + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def find_good_matches(des1, des2, bf, ratio):
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches

def main():
    cap = initialize_video_capture(0)
    my_vid = initialize_video_capture('video.mp4')
    img_target = read_and_resize('Target.jpeg')

    orb = cv2.ORB_create(nfeatures=100000)
    kp1, des1 = orb.detectAndCompute(img_target, None)
    bf = cv2.BFMatcher()

    success, img_video = my_vid.read()
    hT, wT, _ = img_target.shape
    img_video = cv2.resize(img_video, (wT, hT))

    detection = False
    frame_counter = 0

    while True:
        ret_webcam, img_webcam = cap.read()
        if not ret_webcam:
            print("Erro ao capturar da webcam.")
            break

        img_aug = img_webcam.copy()
        kp2, des2 = orb.detectAndCompute(img_webcam, None)

        if not detection:
            my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            if frame_counter >= my_vid.get(cv2.CAP_PROP_FRAME_COUNT):
                my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
            success, img_video = my_vid.read()
            if success:
                img_video = cv2.resize(img_video, (wT, hT))

        img_features = img_webcam.copy()
        if des2 is not None:
            good_matches = find_good_matches(des1, des2, bf, ratio=0.90)
            print(f"Bons matches: {len(good_matches)}")

            img_features = cv2.drawMatches(img_target, kp1, img_webcam, kp2, good_matches, None, flags=2)

            if len(good_matches) > 20:
                detection = True

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

                if matrix is not None:
                    pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)

                    img_warp = cv2.warpPerspective(img_video, matrix, (img_webcam.shape[1], img_webcam.shape[0]))
                    mask_new = np.zeros((img_webcam.shape[0], img_webcam.shape[1]), dtype=np.uint8)
                    cv2.fillPoly(mask_new, [np.int32(dst)], 255)
                    mask_inv = cv2.bitwise_not(mask_new)

                    img_aug = cv2.bitwise_and(img_aug, img_aug, mask=mask_inv)
                    img_aug = cv2.bitwise_or(img_warp, img_aug)

        img_stacked = stack_images(
            [[img_webcam, img_video, img_target],
             [img_features, img_aug, img_aug]], scale=0.5
        )

        cv2.imshow('AR Stack', img_stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    cap.release()
    my_vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()