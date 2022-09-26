import cv2
import os
import re 

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.
    https://stackoverflow.com/a/2669120/11814682
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def images_to_vidoe(video_path, images_files = None, image_folder = None, format="png") :
    """Thanks https://stackoverflow.com/a/44948030/11814682"""

    assert (image_folder is not None) ^ (images_files is not None)

    if images_files is None :
        images_files = [os.path.join(image_folder, img) for img in sorted_nicely(os.listdir(image_folder)) if img.endswith(f".{format}")]
    
    frame = cv2.imread(images_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, 1, (width,height))
    for image in images_files : video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()