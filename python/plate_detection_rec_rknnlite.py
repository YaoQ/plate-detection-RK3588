import cv2
import numpy as np
import argparse
import platform
from utils import COLORS, PLATE_CLASSES 
from plate_rec_rknnlite import PLATE_REC
from rknnlite.api import RKNNLite
from postprocess_numpy import PostProcess
from PIL import Image, ImageDraw, ImageFont

class YOLOv5:
    def __init__(self, args):
        # load model
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(args.det_model)
        if ret != 0:
            print('load rknnlite model failed!')
            exit(ret)

        self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.batch_size = 1
        self.net_h = 640
        self.net_w = 640
        self.agnostic = True
        self.multi_label = True
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.max_det = 1000
        
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )
    
    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )
        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return img, ratio, (tx1, ty1) 
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def predict(self, input_img):
        return self.rknn_lite.inference(inputs=[input_img], data_format=['nchw']) 
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        outputs = self.predict(input_img)
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        det = results[0]

        return det

def extract_and_align_plate(img, det):
    # 提取车牌的4个顶点坐标
    landmarks = det[5:13].reshape(-1, 2).astype(np.int32).tolist()
    pts = np.array(landmarks, dtype=np.float32)

    # 计算车牌的宽高
    width = int(np.linalg.norm(pts[1] - pts[0]))  # 计算左上和右上点之间的距离作为宽度
    height = int(np.linalg.norm(pts[2] - pts[1]))  # 计算右上和右下点之间的距离作为高度
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # 应用透视变换
    aligned_plate = cv2.warpPerspective(img, M, (width, height))

    return aligned_plate

def process_dual_layer_plate(img, det):
    aligned_plate = extract_and_align_plate(img, det)
    
    height, width = aligned_plate.shape[:2]
    top_part = aligned_plate[0:height *5//12, :]
    bottom_part = aligned_plate[height * 1//3:, :]
    top_resized = cv2.resize(top_part, (width, height//2))
    bottom_resized = cv2.resize(bottom_part, (width, height//2))
    aligned_plate = np.hstack((top_resized, bottom_resized))
    return aligned_plate

def draw_results(img, det, plate_rec):
    x1, y1, x2, y2 = det[:4].astype(np.int32).tolist()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    landmarks = det[5:13].reshape(-1, 2).astype(np.int32)
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    font = ImageFont.truetype("../font/NotoSansCJK-Regular.otf", 20, encoding="utf-8")
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1,y1 -30), plate_rec, font = font, fill = (0,0,255))

    return np.array(img_pil)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../imgs/京A25016_32.jpg', help='path of input')
    parser.add_argument('--det_model', type=str, default='../weights/car_plate_detect_1x3x80x80x16-640-640_rm_transpose_rk3588.rknn', help='path of model')
    parser.add_argument('--rec_model', type=str, default='../weights/plate_recognition_color-168-48_rm_transpose_rk3588.rknn', help='path of model')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    
    ## 初始化
    plate_det = YOLOv5(args)
    plate_rec = PLATE_REC(args)

    ## 读取图片
    img = cv2.imread(args.input)
    drawimage = img.copy()
    # 车牌检测
    det_results = plate_det([img])

    for det in det_results:
        x1, y1, x2, y2 = det[:4].astype(np.int32).tolist()
        class_id = int(det[13])
        if class_id == 2:
            continue 
        
        # 提取车牌
        if class_id == 0:
            ## 处理单层车牌
            aligned_plate = extract_and_align_plate(img, det)
        else:
            # 处理双层车牌
            aligned_plate = process_dual_layer_plate(img, det) 

        # 车牌识别
        pred_numer, pred_color = plate_rec([aligned_plate])
        result = str(pred_numer + ", " + pred_color)
        print(result)
        # 车牌识别
        drawimage = draw_results(drawimage, det, result)

    cv2.imwrite("result.jpg", drawimage)
    # 保存识别结果
    print("save result to result.jpg")