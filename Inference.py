import subprocess
import numpy as np
import os

class Inference(): 
    w_img = 416
    h_img = 416
    is_obj_det = True
    def __init__(self, modelName="yolo2", imgfile = './data/dog.jpg', backend="tensorflow", device="CUDA:0"):
        subprocess.call("mkdir onnx", shell=True)
        self.modelName = modelName
        self.imgfile = imgfile
        self.backend = backend
        self.device = device
        #self.predict()
    
    def save_pretrained_model_to_ONNX(self,modelName="yolo2"):
        onnxfilepath = "./onnx/{}.onnx".format(modelName)
        if modelName == "yolo2":
            self.w_img, self.h_img = 416, 416
            cfgfile =  './cfg/yolo.cfg' 
            weightfile =  './yolo.weights'
            # ref: 1.yolo2_pytorch_onnx_save_model.ipynb
            #---chk cfgfile---
            if not os.path.isfile(os.getcwd()+cfgfile[1:]) : print('cfg file Error!')
            #---download weight---
            if not os.path.isfile(os.getcwd()+weightfile[1:]) :
                print('Downloading weights from Web... start')
                subprocess.call("wget http://pjreddie.com/media/files/yolo.weights", shell=True)
                print('Downloading weights from Web... Done!')
            else:
                print('weights file has already exist!')
            #---get model---
            from darknet import Darknet
            m = Darknet(cfgfile)
            m.load_weights(weightfile)
            print('Loading weights from local [%s]... Done!' % (weightfile))
            #---save detection information---
            import pickle
            op_dict = {
                'num_classes':m.num_classes,
                'anchors':m.anchors,
                'num_anchors':m.num_anchors
            }
            pklfilepath = '{}_detection_information.pkl'.format(modelName)
            pickle.dump(op_dict, open(pklfilepath,'wb'))
            #---use Onnx to convert model---
            import torch.onnx
            from torch.autograd import Variable
            dummy_input = Variable(torch.randn(1, 3, self.w_img, self.h_img))# 3 channels, 416*416,
            print('onnx file export [%s]...start' % (onnxfilepath))
            torch.onnx.export(m, dummy_input, onnxfilepath )
            print('onnx file export [%s]... Done!' % (onnxfilepath))
        else: 
            self.is_obj_det = False
            self.w_img, self.h_img = 224, 224
            # ref: 3.vggnet_onnx.ipynb 
            from torch.autograd import Variable
            import torch.onnx
            import torchvision
            if hasattr(torchvision.models, modelName):
                model = getattr(torchvision.models, modelName)(pretrained=True)
                dummy_input = Variable(torch.randn(1, 3, self.w_img, self.h_img))# ImageNet, 3 channels, 224*224,
                torch.onnx.export(model, dummy_input, onnxfilepath )
            else:
                print( "Wrong model name: {}".format(modelName))

    def load_model_from_ONNX(self,modelName="yolo2"):
        import onnx
        # Load the ONNX model
        onnxfilepath = "./onnx/{}.onnx".format(modelName)
        if os.path.isfile(os.getcwd()+onnxfilepath[1:]) :
            print('load onnx file [%s]...start' % (onnxfilepath))
            model = onnx.load(onnxfilepath)
            print('load onnx file [%s]...done!' % (onnxfilepath))
            # Check that the IR is well formed
            onnx.checker.check_model(model)
            # Print a human readable representation of the graph
            model_flat_IR = onnx.helper.printable_graph(model.graph)
            return model, model_flat_IR
        else:
            print('onnx file path Error!')


    def prepare_image(self,imgfile = './data/dog.jpg'):
        from PIL import Image 
        img = Image.open(imgfile).convert('RGB').resize( (self.w_img, self.h_img) )
        img_arr = np.array(img)
        img_arr = np.expand_dims(img_arr, -1)
        img_arr = np.transpose(img_arr, (3,2,0,1))/255.0
        print(img_arr.shape)
        return img, img_arr.astype(np.float64)
        
    def prepare_backend(self, model, framework="tensorflow", device="CUDA:0"): 
        """
        framework = "tensorflow" or "caffe2"
        device = "CUDA:0" or "CPU" 
        """
        print('prepare_backend with [%s] using [%s]...start' % (framework,device))
        if framework == "tensorflow": 
            import onnx_tf.backend as backend
            rep = backend.prepare(model, device=device)   
        elif framework == "caffe2": 
            import onnx_caffe2.backend as backend
            rep = backend.prepare(model, device=device) 
        print('prepare_backend [%s] [%s]...done!' % (framework,device))
        return rep
        
    def detect(self, img, outputs, modelName="yolo2", conf_thresh=0.5, nms_thresh=0.4, output_img_path='predictions.jpg'):
        print('detect ...' )
        #load detection information
        import pickle
        pklfilepath = '{}_detection_information.pkl'.format(modelName)
        detection_information = pickle.load(open(pklfilepath,'rb'))
        num_anchors, anchors, num_classes = [detection_information[k] for k in detection_information.keys()]
        #use original pytorch-yolo2 module to decect outputs
        import torch
        from torch.autograd import Variable
        output = torch.FloatTensor(outputs).cuda() 
        #from utils import *
        from utils import get_region_boxes, nms, load_class_names, plot_boxes
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        for i in range(2):
            boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)[0]
            boxes = nms(boxes, nms_thresh)

        class_names = load_class_names(namesfile)
        str_ = plot_boxes(img, boxes, output_img_path, class_names)        
        print('detect ... done!' )
        return str_
        
    def inference(self, rep, img_arr):
        print('inference ...start' )
        outputs = rep.run(img_arr)
        print(outputs[0].shape)
        print('inference ... done!' )
        return outputs
    
    def predict(self,):
        self.save_pretrained_model_to_ONNX(self.modelName)
        model, model_flat_IR = self.load_model_from_ONNX(self.modelName)
        img, img_arr = self.prepare_image(self.imgfile)
        rep = self.prepare_backend(model, self.backend, self.device)
        outputs = self.inference(rep, img_arr)
        outputs = np.array(outputs).squeeze(0)
        print(outputs.shape)
        if self.is_obj_det:
            str_ = self.detect(img, outputs, self.modelName, conf_thresh=0.5, nms_thresh=0.4, output_img_path='predictions.jpg')
            return str_
        else:
            outputs = np.array(outputs).squeeze(0)
            from imagenet1000_clsid_to_human import cls_dict
            str_ = ""
            for i in reversed(np.argsort(outputs)[-5:]):
                str_ +=  "{:.2f}% : {} \n".format(outputs[i], cls_dict[i])
            return str_
