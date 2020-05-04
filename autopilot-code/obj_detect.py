import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import glob
import time
import imutils

def detect_obj(image, label_out=False):
    """
    Simply detects objects in an image and can return the labels for the objects
    detected if label_out=True
    """
    bbox, label, conf = cv.detect_common_objects(image)
    img_out = draw_bbox(image, bbox, label, conf, write_conf=True)
    if label_out == True:
        return (img_out, label)
    else:
        return img_out

def detect_multi_obj(path, output_path):
    """
    Reads in images from a folder and detects all objects in the images and saves
    to a specified directory
    """
    a = 0
    data_path = os.path.join(path, "*g")
    files = glob.glob(data_path)
    for f1 in files:
        a += 1
        img = cv2.imread(f1)
        img_obj = detect_obj(img)
        
        filename = os.path.basename(f1)
        cv2.imwrite(output_path + filename, img_obj)
    print("Processed {} images" .format(a))
        

def process_obj(path, show_obj=False, save_img=False, save_dir=None):
    """
    Processess all images in a given folder and detects objects on all the images
    returns the number of 
    """
    num_obj = 0
    num_label = 0 
    
    data_path = os.path.join(path, "*g")
    files = glob.glob(data_path)
    
    for f1 in files:
        start = time.time()
        labels = []
        filename = os.path.basename(f1)
        
        img = cv2.imread(f1)
        img_obj, labels = detect_obj(img, label_out=True)
        if labels:
            num_obj += 1
            num_label += len(labels)
            
        if show_obj == True:
            cv2.imshow("Object Detection", img_obj)
            cv2.waitKey(1)
        
        if save_dir != None:
            cv2.imwrite(save_dir + filename, img_obj)
        
        print("Time elapsed: {}." .format(time.time() - start))
            
    cv2.destroyAllWindows()
    return(num_obj, num_label)

#%%
#Focal length calculation 
#NOW REDUNDENT
def find_marker(img):
    """
    Finds the bounding box for a detected object and returns a list of the coordinates
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    return (cv2.minAreaRect(c))

def focal_approx(image, known_dis, known_dim):
    """
    Use this one time to find the focal length of the camera being used to take
    the image.
    """
    marker = find_marker(image)
    obj_w = known_dim[0]
    focal = (marker[1][0] * known_dis) / obj_w
    return focal

"""
focal length calc:
    simulation:
        length= 0.5316919
        width = 0.5316918
        height = 0.531692
        distance = 0.846957
    
focal length ~495.0, =494.9705467

copy this into terminal:
    focal = 494.97054669950995
"""