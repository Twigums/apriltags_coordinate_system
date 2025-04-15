import os, cv2, toml
import numpy as np
from opencv_functions import get_camera_calibration, undistort_image, rotate_image

PATH_TO_PICTURES_FOLDER = "./pictures"
PATH_TO_BBOX_FOLDER = "./bbox"
PATH_TO_DESTINATION_FOLDER = "./templates"


def rotate_gui(image):
    angle = 0
    
    window_name = "Rotate Image"
    cv2.namedWindow(window_name)

    while True:
        rotated_image = rotate_image(image, angle)
        rotated_image_copy = rotated_image.copy()

        cv2.imshow(window_name, rotated_image_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == 81: # left arrow on linux
            angle -= 0.5

        elif key == 83: # right arrow on linux
            angle += 0.5

        elif key == ord("s"):
            cv2.destroyAllWindows()
            
            return rotated_image, angle

    cv2.destroyAllWindows()
        
    return None

def crop_gui(image):
    x, y, width, height = cv2.selectROI("Do the intern's job!!", image, fromCenter = False, showCrosshair = True)
    bbox_dict = {
        "bbox": {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    }
    
    cropped_image = image[y:y + height, x:x + width]
    cv2.destroyAllWindows()
    
    return cropped_image, bbox_dict

def main():
    all_pictures = [f for f in os.listdir(PATH_TO_PICTURES_FOLDER) if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    
    for path_to_picture in all_pictures:
        path_to_bbox = os.path.join(PATH_TO_BBOX_FOLDER, path_to_picture)[:-3] + "toml"
        path_to_output = os.path.join(PATH_TO_DESTINATION_FOLDER, "roi_" + path_to_picture)
        
        if not os.path.exists(path_to_bbox):
            image = cv2.imread(os.path.join(PATH_TO_PICTURES_FOLDER, path_to_picture))
            rotated_image, angle = rotate_gui(image)
            res_image, bbox_dict = crop_gui(rotated_image)

            bbox_dict["angle"] = angle
            with open(path_to_bbox, 'w') as f:
                toml.dump(bbox_dict, f)

            cv2.imwrite(path_to_output, res_image)

if __name__ == "__main__":
    main()