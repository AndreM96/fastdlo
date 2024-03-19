import os, cv2
from fastdlo.core import Pipeline

from utility_rs import *

if __name__ == "__main__":

    test = False

    ######################
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"
    IMG_W = 640
    IMG_H = 360

    if test:
        IMG_PATH = "test_images\9.jpg"
        # COLOR
        source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    else:
        profile, pipeline,hole_filling0,align,intr,geometrie_added, depth_scale  = initial_configuration()
        depth_image, depth_frame, rgb_image = get_frame(pipeline, hole_filling0, align)
        # COLOR
        source_img = cv2.resize(rgb_image, (IMG_W, IMG_H))

    ######################

    script_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_siam = os.path.join(script_path, "weights/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "weights/" + ckpt_seg_name)
    

    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)


    img_out, _ = p.run(source_img=source_img, mask_th=77)

    canvas = source_img.copy()
    canvas = cv2.addWeighted(canvas, 1.0, img_out, 0.8, 0.0)
    cv2.imshow("output", canvas)
    cv2.waitKey(0)

