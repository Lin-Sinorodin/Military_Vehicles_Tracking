import os
import cv2
from tqdm.autonotebook import tqdm
from cv2.dnn_superres import DnnSuperResImpl_create


def get_model_weights(models_path:str):
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
        
        for i in [2, 3, 4]:
            model_url = f'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x{i}.pb'
            os.system(f'wget -P {models_path} {model_url}')


def initialize_model(scale:int, models_path:str):
    sr_model = DnnSuperResImpl_create()
    sr_model.readModel(f'{models_path}/ESPCN_x{scale}.pb')
    sr_model.setModel("espcn", scale)
    return sr_model


models_path = 'super_resolution_models'

get_model_weights(models_path)

sr_x2 = initialize_model(2, models_path)
sr_x3 = initialize_model(3, models_path)
sr_x4 = initialize_model(4, models_path)



def on_image(img, scale:int):
    """Perform super resolution on an image with scale of 2/3/4"""
    if scale == 2:
        return sr_x2.upsample(img)
    elif scale == 3:
        return sr_x3.upsample(img)
    elif scale == 4:
        return sr_x4.upsample(img)
    else:
        raise Exception("Scale should be 2, 3 or 4")

        
def on_folder(images_dir:str, new_images_dir:str, scale:int):
    """Perform super resolution on a folder with scale of 2/3/4"""
    os.makedirs(new_images_dir, exist_ok=True)

    images_files = sorted(os.listdir(images_dir))
    for img_file in tqdm(images_files):
        img_path = f'{images_dir}/{img_file}'
        img = cv2.imread(img_path)
        img_resized = on_image(img, scale)
        cv2.imwrite(f'{new_images_dir}/{img_file}', img_resized)
