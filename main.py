from efficientdet_test import out,invert_affine

import cv2
import numpy as np

def crop_bbox_from_image(image, bbox):
    x1, y1, x2, y2 = bbox
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

if __name__=="__main__":
    img_path = "./test/cows.jpg"
    print(out)
    preds = invert_affine(framed_metas, out)
    # 假设 preds 是模型预测的结果，其中 preds[i]['rois'][j] 是第 i 张图像的第 j 个边界框的坐标
    for i in range(len(preds)):
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            cropped_img = crop_bbox_from_image(ori_imgs[i], (x1, y1, x2, y2))

            # 处理 cropped_img，例如保存到文件或进一步处理
            cv2.imwrite(f'cropped_image_{i}_{j}.jpg', cropped_img)
