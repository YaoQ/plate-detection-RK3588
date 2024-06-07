import cv2
import numpy as np
import argparse
import platform
from utils import COLORS, PLATE_CLASSES 
from rknnlite.api import RKNNLite

class PLATE_REC:
    def __init__(self, args):
        # load model
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(args.rec_model)
        if ret != 0:
            print('load rknnlite model failed!')
            exit(ret)

        self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        self.color=['黑色','蓝色','绿色','白色','黄色'] 
        self.batch_size = 1
        self.net_h = 48
        self.net_w = 168

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        im = cv2.resize(ori_img, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        img = im.transpose((2, 0, 1))  # HWC to CHW
        return img

    def decodePlate(self, preds):
        # 将 preds 转换为 Python 列表
        preds = preds.flatten().tolist()
    
        # 使用列表推导式和 zip 函数来过滤重复和零值
        newPreds = [preds[i] for i in range(len(preds)) if preds[i] != 0 and (i == 0 or preds[i] != preds[i-1])]
    
        # 使用 ''.join() 来拼接字符串
        plate = ''.join([self.palteStr[i] for i in newPreds])
    
        return plate

    def predict(self, input_img):
        return self.rknn_lite.inference(inputs=[input_img], data_format=['nchw']) 
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        preds, preds_color  = self.predict(input_img)

        max_values = np.argmax(preds, axis=-1)
        preds_number = self.decodePlate(max_values)
        preds_color=preds_color.argmax()
        color=self.color[preds_color.item()]

        return preds_number, color

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../imgs/test.jpg', help='path of input')
    parser.add_argument('--rec_model', type=str, default='../weights/plate_recognition_color-168-48_rm_transpose_rk3588.rknn', help='path of model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    plate_rec_result =PLATE_REC(args)
    preds_number, color = plate_rec_result([cv2.imread(args.input)])
    print(preds_number, color, args.input )
