import cv2
from PIL import Image
import argparse
import re
import time

print("cv2 version : ", cv2.__version__)

# Input Image and Select SR Model
args = argparse.ArgumentParser()
args.add_argument('--path', '-p', help='image path', type=str)
args.add_argument('--model', '-m', help='super resolution model', type=str)
pars = args.parse_args()

imagePath = pars.path
imageModel = pars.model

# Time Checking
startime = time.time()

print("Loading Image...")

# Read Image
img = cv2.imread(imagePath)
# plt.imshow(img[:,:,::-1])
print("Your Image Size : ", img.shape)

def SR_model_selection(model):

    model_list = ['edsr', 'lapsrn', 'espcn', 'fsrcnn', 'fsrcnn-small']

    model_x = re.findall(r'\d+', model)
    model_x = model_x[0]

    model_name = model.lower()
    idx = model_name.find('_')
    model_name = model_name[:idx]

    try:

        if model_name in model_list:

            idx = model_list.index(model_name)

            if idx == 0:
                ModelPB = "EDSR_x"+str(model_x)+".pb"
            elif idx == 1:
                ModelPB = "LapSRN_x"+str(model_x)+".pb"
            elif idx == 2:
                ModelPB = "ESPCN_x" + str(model_x) + ".pb"
            elif idx == 3:
                ModelPB = "FSRCNN_x" + str(model_x) + ".pb"
            elif idx == 4:
                ModelPB = "FSRCNN-small_x" + str(model_x) + ".pb"

        # dnn super resolution_opencv
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # .pb파일 path
        load_model = "./cv2_dnn_sr/" + ModelPB
        sr.readModel(load_model)

        # set model
        sr.setModel(model_name, int(model_x))
        result = sr.upsample(img)

        # ndarray to PIL image
        pil_image = Image.fromarray(result[:, :, ::-1])
        ModelPB = re.sub('.pb', '', ModelPB)
        pil_image.save(ModelPB+'_result.jpeg')

        print("Original Image : ", img.shape, ",  Super-Resolution Image : ", result.shape)
        print("Successfully saved!!!")

        endtime = time.time()
        print("Total Duration : ", endtime - startime)

    except Exception as e:
        print("There's no model.")


SR_model_selection(imageModel)










