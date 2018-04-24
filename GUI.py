
# coding: utf-8

from time import time, ctime, localtime

#python version
from sys import version
print('Python version:',version)

#import ttk
if version[0] == '2':
    from Tkinter import *
    from ttk import Button, Checkbutton, Label, Entry, Combobox
else: #version[0] == '3':
    from tkinter import *
    from tkinter.ttk import Button, Checkbutton, Label, Entry, Combobox

from PIL import ImageTk, Image
#from subprocess import call
from Inference import Inference
models=['yolo2', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
backends =  ['tensorflow', 'caffe2']
devices =  ["CPU" , "CUDA:0"]
#from IR_Extraction import get_list_of_sequencial_nodes
#options_for_seq_search = get_list_of_sequencial_nodes()
options_for_seq_search = list(reversed( [['Conv'], 
                                        ['Conv', 'Add'], 
                                        ['Conv', 'Add', 'Relu'], 
                                        ['Conv', 'Add', 'Relu', 'MaxPool'], 
                                        ['Conv', 'BatchNormalization'],
                                        ['Conv', 'BatchNormalization', 'LeakyRelu'],
                                        ['Conv', 'BatchNormalization', 'LeakyRelu', 'MaxPool']]))
seq_search_TXT = ["-> ".join(seq) for seq in options_for_seq_search] 

class GUI():
    
    vb_dict = {}
    infer = None 
    modelName,backend,device = "", "", ""
    def __init__(self, master):
        self.master = master 
        self.row = 0 #for grid
        self.all_comp = []
        self.get_comp = []
        self.vb_name = []  
        self.buildGUI_1()
        #self.show_all_variable()
        
    # --------Mode interface ---------    
    def buildGUI_1(self): #CNN
        self.master.title('build CNN -- load model')         
        self.label_1to1_text_combobox("Model", models , width=50 ) # models = ("inception_v3","vgg16"...)
        self.label_1to1_text_entry(name="ImageUrl", default_text="data/person.jpg", width=100)
        self.label_1to1_text_combobox("Backend", backends , width=50 ) 
        self.label_1to1_text_combobox("Device", devices , width=50 ) 
        self.label_1to1_text_combobox("SearchSeq", seq_search_TXT , width=100 )
        self.vb_dict = self.generate_variable_dict()
        self.button(self.click_show_orig_img, "Show Orig Image")
        self.button(self.click_inference, "Inference")
        self.button(self.click_show_model, "Model Visualization")
        self.button(self.click_search_sequencial_nodes, "Search Nodes")
        self.show_str = StringVar()
        self.textvariable_label(textvariable=self.show_str)
        
    def click_show_orig_img(self):
        self.vb_dict = self.generate_variable_dict() # cannot skip
        img_path = self.vb_dict["ImageUrl"]
        self.img_label(path=img_path,title='Original image')
        
    def click_show_model(self):
        #build infer
        self.build_infer()
        show_str = "\nmodel name = {} \n\nnumber of param = {} \n\nnumber of flops = {}\n\n ".format(self.infer.modelName , self.infer.n_param, self.infer.n_flops)
        import webbrowser
        webbrowser.open(self.infer.svgfilepath)  # open <svgfilepath> in web
        self.show_str.set(show_str)#show txt
        
    def build_infer(self):
        self.show_str.set(" \n\n\n!!!!! ")#clear old txt
        #self.show_str.set("Please wait....building model")#show txt
        self.vb_dict = self.generate_variable_dict() # cannot skip
        modelName = self.vb_dict["Model"]
        backend = self.vb_dict["Backend"]
        device = self.vb_dict["Device"]
        is_chg = not (self.modelName==modelName and self.backend==backend and self.device==device) 
        if (self.infer is None) or (is_chg): 
            self.infer = Inference(modelName=modelName,backend=backend, device=device)
            self.modelName=modelName; self.backend=backend; self.device=device
        
    def click_search_sequencial_nodes(self):
        #build infer
        self.build_infer()
        #seq: ["Conv", "Add", ...]
        self.vb_dict = self.generate_variable_dict() # cannot skip
        seqTXT = self.vb_dict["SearchSeq"]
        seq = options_for_seq_search[seq_search_TXT.index(seqTXT)]
        #search_n_visualize_sequence
        #self.show_str.set("Please wait....search_sequencial_nodes")#show txt
        show_str, is_match, marked_svgfilepath = self.infer.search_n_visualize_sequence(seq)
        self.show_str.set(str(show_str))#show txt
        if is_match:
            import webbrowser
            webbrowser.open(marked_svgfilepath)  # open <svgfilepath> in web
            
    def click_inference(self):
        #build infer
        self.build_infer()
        #img
        self.vb_dict = self.generate_variable_dict() # cannot skip
        imgfile = self.vb_dict["ImageUrl"]
        #predict
        #self.show_str.set("Please wait....predicting")#show txt
        str_, time_cost = self.infer.predict(imgfile=imgfile)
        str_ = "Time cost : {} \n\n".format(time_cost) + str_ 
        self.show_str.set(str(str_))#show txt
        #pred img
        if self.infer.is_obj_det:# if is Object Detect  
            self.img_label(path="predictions_samesize.jpg",title='Prediction image')   
        
    # --------Component Conbination --------- 
    def label_1to1_text_combobox(self, name="", values=("1","2"), default_Chosen=0, width=10):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(StringVar(self.master, values[default_Chosen]))
        self.all_comp.append(Combobox(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1]['values'] = values
        self.all_comp[-1].current(default_Chosen) 
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_float_entry(self, name="", default_float=0.01, width=5):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(DoubleVar(self.master, default_float))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_int_entry(self, name="", default_int=-1, width=10):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w')
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(IntVar(self.master, default_int))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def label_1to1_text_entry(self, name="", default_text="", width=35):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name)) 
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(StringVar(self.master, default_text))
        self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1])) 
        #self.all_comp[-1].pack()             
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
            
    def label_1to3_int_entry(self, name="", default_int=[-1,-1,-1], width=10):
        for i in [1,2,3]:
            self.vb_name.append(name.split()[0]+str(i))
        self.all_comp.append(Label(self.master, text=name))
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        for i, int_ in enumerate(default_int):
            self.get_comp.append(IntVar(self.master, int_))
            self.all_comp.append(Entry(self.master, width=width, textvariable=self.get_comp[-1]))
            #self.all_comp[-1].pack() 
            self.all_comp[-1].grid(row=self.row, column=i+1, sticky=W) #  
        self.row += 1
            
    def label_1to1_bool_checkbutton(self, name="", default_bool=True ,default_text="Yes"):
        self.vb_name.append(name.split()[0])
        self.all_comp.append(Label(self.master, text=name))
        #self.all_comp[-1].pack(anchor='w') 
        self.all_comp[-1].grid(row=self.row, sticky=E) #
        self.get_comp.append(BooleanVar(self.master, default_bool))
        self.all_comp.append(Checkbutton(self.master, text=default_text, variable =self.get_comp[-1], offvalue =False, onvalue =True)); 
        #self.all_comp[-1].pack() 
        self.all_comp[-1].grid(row=self.row, column=1, sticky=W) #
        self.row += 1 #
        
    def textvariable_label(self, textvariable):        
        self.all_comp.append(Label(self.master, textvariable=textvariable))         
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #
        
    def text_label(self, text="OK"):        
        self.all_comp.append(Label(self.master, text=text))         
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #
        
    def text_text(self, text="OK"):        
        text_ = Text(self.master, width=80 , height=10)
        text_.insert(INSERT, text)
        text_.insert(END, "")
        self.all_comp.append(text_) 
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #   

    def img_label(self, path="cat1.jpeg", title='image'): 
        window = Toplevel()
        #window.geometry('400x400')
        window.title(title)  
        img = ImageTk.PhotoImage(Image.open(path))
        label = Label(window, image = img)
        label.pack()
        window.mainloop()
        
    def button(self, command, text="OK"):        
        self.all_comp.append(Button(self.master, text=text, command = command))         
        #self.all_comp[-1].pack()
        self.all_comp[-1].grid(row=self.row, column=1) #
        self.row += 1 #
        
    # ---Show------------------------------------------------------    
    def show_all_variable(self):
        print('show all variable:')
        i=0
        for c in self.get_comp:
            print('{} = {}'.format(self.vb_name[i],c.get()))
            i+=1
              
    def generate_variable_dict(self):
        i=0
        d={}
        for c in self.get_comp:
            d.update({self.vb_name[i]:c.get()})
            i+=1  
        return d   



root = Tk()
my_gui = GUI(root)
root.mainloop() 
    


