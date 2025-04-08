import apriltag, cv2
import numpy as np


def apriltag_init() -> apriltag.Detector:
    options = apriltag.DetectorOptions(
        families='tag36h11',
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True)

    detector = apriltag.Detector()

    return detector

def detect_apriltag(detector: apriltag.Detector, img: np.ndarray) -> list:
    res = detector.detect(img)

    if len(res) > 0:
        # print(res)
        pass

    return res