import numpy as np
from Onnx import OnnxImportExport, prepare_backend
import pickle
import time
from IR_Extraction import *
from PIL import Image 

class Inference(): 
    
    # define in self.prepare_image()
    w_orig_img, h_orig_img =  -1, -1
    # define in self.prepare_model_and_backend()
    is_obj_det, w_img, h_img = True, -1, -1
    model, model_flat_IR = None, None
    rep, backend = None, None
    prepare_time = 0.0
    # define in self.extract_IR_info()
    svgfilepath = ""
    Node2nextEntity, Entity2nextNode = None, None
    n_param, n_flops = -1, -1
    #options_for_sequence_search = []
    
    def __init__(self, modelName="yolo2", backend="tensorflow", device="CUDA:0"):
        self.modelName = modelName
        self.backend = backend
        self.device = device
        self.prepare_model_and_backend()
        self.extract_IR_info()
        #self.predict()
        #slef.search_n_visualize_sequence(search_sequence=['Conv', 'Add', 'Relu', 'MaxPool'])
    
    def prepare_model_and_backend(self,):
        print("Inference: Prepare model and backend...start");st = time.time()
        ONNX = OnnxImportExport()
        self.is_obj_det, self.w_img, self.h_img = ONNX.save_pretrained_model_to_ONNX(self.modelName)
        self.model, self.model_flat_IR = ONNX.load_model_from_ONNX(self.modelName)
        
        self.rep, self.backend = prepare_backend(self.model, self.backend, self.device)
        self.prepare_time = time.time()-st
        print("Inference: Prepare model and backend...end, {:.2f} sec".format(self.prepare_time))        
        
    
    def extract_IR_info(self,): 
        """
        model, rep is required
        """
        
        # n_param, n_flops
        """
        #20180615 modified: 
        init_shape_dict = self.extract_init_shape_dict_from_IR(self.rep, self.modelName)
        overall_shape_dict = self.extract_get_overall_shape_dict_from_IR(self.model, 
                                                                         init_shape_dict, 
                                                                         self.backend, 
                                                                         self.modelName)
        kernel_shape_dict = self.extract_kernel_shape_dict_from_IR(self.model, overall_shape_dict, 
                                                              self.backend, self.modelName)
        self.n_param, self.n_flops = calculate_num_param_n_num_flops(kernel_shape_dict)
        """
        #20180615 modified: ignore save to pickle
        init_shape_dict = get_init_shape_dict(self.rep)
        overall_shape_dict = init_shape_dict
        kernel_shape_dict = get_kernel_shape_dict(self.model, overall_shape_dict)
        self.n_param, self.n_flops = calculate_num_param_n_num_flops(kernel_shape_dict)
                
        #visualization
        self.svgfilepath = generate_svg(self.modelName)
        
        #graph related
        self.Node2nextEntity, self.Entity2nextNode = self.extract_graph_edges_dict_from_IR(self.model, self.modelName)
        #self.options_for_sequence_search = get_list_of_sequencial_nodes()

    def search_n_visualize_sequence(self, search_sequence=['Conv', 'Add', 'Relu', 'MaxPool'], if_print = False): 
        #search_sequence
        matching_nodes = find_sequencial_nodes(self.model, 
                                               self.Node2nextEntity, 
                                               self.Entity2nextNode, 
                                               search_sequence, 
                                               if_print=if_print)
        if matching_nodes == []:
            is_match = 0
            show_str = "\nsearch: \n{}, \nget matching node: \n{}\n\n".format(search_sequence, "NOT FOUND!!")
        else:
            is_match = 1
            show_str = "\nsearch: \n{}, \nget matching node: \n{}\n\n".format(search_sequence, matching_nodes )
        print(show_str)
        
        #draw SVG
        marked_svgfilepath = ""
        if is_match:
            from IR_Extraction import generate_svg
            marked_svgfilepath = generate_svg(self.modelName, marked_nodes=matching_nodes)
        
        return show_str, is_match, marked_svgfilepath 
        
        
    def predict(self, imgfile = './data/dog.jpg'):
        """
        model, rep is required
        """
        img, img_arr = self.prepare_image(imgfile, (self.w_img, self.h_img))
        
        #inference
        print("Inference: Inference...start");st = time.time()
        outputs = self.inference(self.rep, img_arr)
        outputs = np.array(outputs).squeeze(0)
        #print(outputs.shape)#check outputs shape
        inference_time = time.time()-st
        print("Inference: Inference...end, {:.2f} sec".format(inference_time))
        
        #gen txt
        time_cost = "prepare_time: {:.2f} sec, inference_time: {:.2f} sec".format(self.prepare_time,inference_time)
        if self.is_obj_det: # Object Detect 
            str_ = self.detect(img, outputs, self.modelName, conf_thresh=0.5, nms_thresh=0.4, output_img_path='predictions.jpg')
            self.resize_prediction_image(savename='predictions_samesize.jpg')
        else:               # Image Classification
            outputs = np.array(outputs).squeeze(0)
            from imagenet1000_clsid_to_human import cls_dict
            str_ = ""
            for i in reversed(np.argsort(outputs)[-5:]):
                str_ +=  "{:.2f}% : {} \n".format(outputs[i], cls_dict[i])
        return str_, time_cost    
    
    def prepare_image(self,imgfile = './data/dog.jpg', resize_shape=(416,416) ):
        #from PIL import Image 
        img = Image.open(imgfile).convert('RGB')
        self.w_orig_img, self.h_orig_img = np.array(img).shape[1], np.array(img).shape[0]
        img_arr = np.array(img.resize(resize_shape))
        img_arr = np.expand_dims(img_arr, -1)
        img_arr = np.transpose(img_arr, (3,2,0,1))/255.0
        #print(img_arr.shape)
        return img, img_arr.astype(np.float64)
        
    def resize_prediction_image(self,imgfile = 'predictions.jpg', savename='predictions_samesize.jpg' ):
        #from PIL import Image 
        orig_img_size = (self.w_orig_img, self.h_orig_img)
        img = Image.open(imgfile).convert('RGB').resize(orig_img_size)
        img.save(savename)
    
    def detect(self, img, outputs, modelName="yolo2", conf_thresh=0.5, nms_thresh=0.4, output_img_path='predictions.jpg'):
        print('Detect ...start' )
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
        print('Detect ... done!' )
        return str_
        
    def inference(self, rep, img_arr):
        print('Inference ...start' )
        outputs = rep.run(img_arr)
        #print(outputs[0].shape) #check outputs shape
        print('Inference ... done!' )
        return outputs
        
        
    def extract_init_shape_dict_from_IR(self, rep, modelName="yolo2" ): 
        """
        save out shape_dict from IR
        """
        saveName = 'onnx/{}_init_shape_dict.pkl'.format(modelName)
        if os.path.isfile(os.getcwd()+"/"+saveName):
            init_shape_dict = pickle.load(open(saveName,'rb'))
        else:
            init_shape_dict = get_init_shape_dict(rep)
            pickle.dump(init_shape_dict,open(saveName,'wb')) #save  
        return init_shape_dict
    
    def extract_get_overall_shape_dict_from_IR(self, model, init_shape_dict, backend, modelName="yolo2"): 
        """
        save out shape_dict from IR
        """        
        saveName = 'onnx/{}_overall_shape_dict.pkl'.format(modelName)
        if os.path.isfile(os.getcwd()+"/"+saveName):
            overall_shape_dict = pickle.load(open(saveName,'rb'))
        else:
            print("get_overall_shape_dict...start");st=time.time()
            overall_shape_dict = get_overall_shape_dict(model, init_shape_dict, backend) 
            print("get_overall_shape_dict...end, {:.2f}".format(time.time()-st))
            pickle.dump(overall_shape_dict,open(saveName,'wb')) #save 
        return overall_shape_dict
    
    def extract_kernel_shape_dict_from_IR(self, model, overall_shape_dict, backend, modelName="yolo2"): 
        """
        save out shape_dict from IR
        """
        saveName = 'onnx/{}_kernel_shape_dict.pkl'.format(modelName)
        if os.path.isfile(os.getcwd()+"/"+saveName):
            kernel_shape_dict  = pickle.load(open(saveName,'rb'))
        else:
            kernel_shape_dict = get_kernel_shape_dict(model, overall_shape_dict)
            pickle.dump(kernel_shape_dict,open(saveName,'wb')) #save 
        return kernel_shape_dict
    
    def extract_graph_edges_dict_from_IR(self, model, modelName="yolo2"): 
        """
        save out shape_dict from IR
        """
        saveName1 = 'onnx/{}_Node2nextEntity_dict.pkl'.format(modelName)
        saveName2 = 'onnx/{}_Entity2nextNode_dict.pkl'.format(modelName)
        if os.path.isfile(os.getcwd()+"/"+saveName1) and os.path.isfile(os.getcwd()+"/"+saveName2):
            Node2nextEntity = pickle.load(open(saveName1,'rb'))
            Entity2nextNode = pickle.load(open(saveName2,'rb'))
        else:
            Node2nextEntity, Entity2nextNode = get_graph_order(model)
            pickle.dump(Node2nextEntity,open(saveName1,'wb'))
            pickle.dump(Entity2nextNode,open(saveName2,'wb'))
        return Node2nextEntity, Entity2nextNode
    

