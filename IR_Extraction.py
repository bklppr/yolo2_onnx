from Onnx import make_dir, OnnxImportExport
import subprocess
import pickle
import os
import numpy as np
import time


        
def generate_svg(modelName):
    """
    generate SVG figure from existed ONNX file
    """
    onnxfilepath = "onnx/{}.onnx".format(modelName)
    dotfilepath = "dot/{}.dot".format(modelName)
    svgfilepath = "svg/{}.svg".format(modelName)
    # check if onnx file exist
    if not os.path.isfile(os.getcwd()+"/"+onnxfilepath):
        print('generate_svg Error! Onnx file not exist!')
        return
    else:
        make_dir("dot")
        make_dir("svg")
        subprocess.call("python net_drawer.py --input {} --output {} --embed_docstring".format(onnxfilepath,dotfilepath), shell=True) # onnx -> dot
        subprocess.call("dot -Tsvg {} -o {}".format(dotfilepath,svgfilepath), shell=True)# dot -> svg
        print('generate_svg success..')
        
def get_init_shape_dict(rep):
    """
    Extract Shape of Initial Input Object
    e.g.
    if
      %2[FLOAT, 64x3x3x3]
      %3[FLOAT, 64]
    then
       return {u'2':(64,3,3,3),u'3':(64,)}
    """
    d = {}
    for key in rep.input_dict:
        tensor = rep.input_dict[key]
        shape = np.array(tensor.shape, dtype=int)
        d.update({key:shape})
    return d    

def get_output_shape_of_node(node, shape_dict, backend, device = "CPU"):# or "CUDA:0"
    """
    generate output_shape of a NODE
    """    
    out_idx = node.output[0]
    input_list = node.input # e.g. ['1', '2']
    
    inps = []
    for inp_idx in input_list:
        inp_shape = shape_dict[inp_idx] 
        rand_inp = np.random.random(size=inp_shape).astype('float16')
        inps.append(rand_inp)
    try:
        out = backend.run_node(node=node, inputs=inps, device=device)
        out_shape = out[0].shape 
    except:
        out_shape = shape_dict[input_list[0]]
        print("Op: [{}] run_node error! return inp_shape as out_shape".format(node.op_type))
        
    return out_shape, out_idx 

def get_overall_shape_dict(model, init_shape_dict, backend):
    """
    generate output_shape of a MODEL GRAPH
    """ 
    shape_dict = init_shape_dict.copy()
    for i, node in enumerate(model.graph.node):
        st=time.time()
        out_shape, out_idx = get_output_shape_of_node(node, shape_dict, backend)
        shape_dict.update({out_idx:out_shape})
        print("out_shape: {} for Obj[{}], node [{}][{}]...{:.2f} sec".format(out_shape, out_idx, i, node.op_type,time.time()-st))
    return shape_dict 

def get_graph_order(model):
    """
    Find Edges (each link) in MODEL GRAPH
    """
    Node2nextEntity = {}
    Entity2nextNode = {} 
    for Node_idx, node in enumerate(model.graph.node):
        # node input
        for Entity_idx in node.input:
            if not Entity_idx in Entity2nextNode.keys():
                Entity2nextNode.update({Entity_idx:Node_idx})
        # node output
        for Entity_idx in node.output:
            if not Node_idx in Node2nextEntity.keys():
                Node2nextEntity.update({Node_idx:Entity_idx}) 
    return Node2nextEntity, Entity2nextNode

def get_kernel_shape_dict(model, overall_shape_dict):
    """
    Get Input/Output/Kernel Shape for Conv in MODEL GRAPH
    """
    conv_d = {}
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = np.array(attr.ints, dtype=int)
                    break
            inp_idx = node.input[0]
            out_idx = node.output[0]
            inp_shape = overall_shape_dict[inp_idx]
            out_shape = overall_shape_dict[out_idx]
            conv_d.update({i:(inp_idx, out_idx, inp_shape, out_shape, kernel_shape)})
            print("for node [{}][{}]:\ninp_shape: {} from obj[{}], \nout_shape: {} from obj[{}], \nkernel_shape: {} \n"
                  .format(i, node.op_type, inp_shape, inp_idx, out_shape, out_idx, kernel_shape ))
    return conv_d

def calculate_num_param_n_num_flops(conv_d):
    """
    calculate num_param and num_flops from conv_d
    """
    n_param = 0
    n_flops = 0
    for k in conv_d:
        #i:(inp_idx, out_idx, inp_shape, out_shape, kernel_shape)
        inp_shape, out_shape, kernel_shape = conv_d[k][2],conv_d[k][3],conv_d[k][4]
        h,w,c,n,H,W = kernel_shape[1], kernel_shape[1], inp_shape[1], out_shape[1], out_shape[2], out_shape[3]
        n_param  += n*(h*w*c+1)
        n_flops  += H*W*n*(h*w*c+1)
    return n_param, n_flops

def find_sequencial_nodes(model, Node2nextEntity, Entity2nextNode, search_target=['Conv', 'Add', 'Relu', 'MaxPool'], if_print = False): 
    """
    Search Where is Subgroup
    """
    found_nodes = []
    for i, node in enumerate(model.graph.node): 
        if if_print: print("\nnode[{}] ...".format(i))
        n_idx = i #init
        is_fit = True
        for tar in search_target:
            try:
                assert model.graph.node[n_idx].op_type == tar #check this node
                if if_print: print("node[{}] fit op_type [{}]".format(n_idx, tar))
                e_idx = Node2nextEntity[n_idx] #find next Entity
                n_idx = Entity2nextNode[e_idx] #find next Node
                #if if_print: print(e_idx,n_idx)
            except: 
                is_fit = False
                if if_print: print("node[{}] doesn't fit op_type [{}]".format(n_idx, tar))
                break

        if is_fit:
            if if_print: print("node[{}] ...fit!".format(i))
            found_nodes.append(i)
        else:
            if if_print: print("node[{}] ...NOT fit!".format(i))
    if if_print: print("\nNode{} fit the matching pattern".format(found_nodes))
    return found_nodes

def get_permutations(a):
    """
    get all permutations of list a
    """
    import itertools
    p = []
    for r in range(len(a)+1):
        c = list(itertools.combinations(a,r))
        
        for cc in c:
            p += list(itertools.permutations(cc))
    return p 

def get_list_of_sequencial_nodes(search_head = ['Conv'], followings = ['Add', 'Relu', 'MaxPool']):
    """
    if 
        search_head = ['Conv']
        followings = ['Add', 'Relu', 'MaxPool']
    return
        [['Conv'],
         ['Conv', 'Add'],
         ['Conv', 'Relu'],
         ['Conv', 'MaxPool'],
         ['Conv', 'Add', 'Relu'],
         ['Conv', 'Relu', 'Add'],
         ['Conv', 'Add', 'MaxPool'],
         ['Conv', 'MaxPool', 'Add'],
         ['Conv', 'Relu', 'MaxPool'],
         ['Conv', 'MaxPool', 'Relu'],
         ['Conv', 'Add', 'Relu', 'MaxPool'],
         ['Conv', 'Add', 'MaxPool', 'Relu'],
         ['Conv', 'Relu', 'Add', 'MaxPool'],
         ['Conv', 'Relu', 'MaxPool', 'Add'],
         ['Conv', 'MaxPool', 'Add', 'Relu'],
         ['Conv', 'MaxPool', 'Relu', 'Add']]
    """
    search_targets = [ search_head+list(foll) for foll in get_permutations(followings)] 
    return search_targets


        
        
        
    