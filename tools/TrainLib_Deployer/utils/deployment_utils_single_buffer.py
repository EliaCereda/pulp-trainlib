'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini
'''

import os
import shutil
import math

from torch import mm
import utils.GM_templates as Gtemp
import utils.net_templates_single_buffer as ntemp


"""
DNN Size Checker backend functions
"""
def max_input_dim(layers_l, cin_l, hin_l, win_l):
    RES = 0
    for layer in range(len(layers_l)):
        temp = cin_l[layer]*hin_l[layer]*win_l[layer]
        if temp > RES:
            RES = temp

    print(f"\nMax data dim: {RES}\n")
    return RES

def max_wgt_dim(layers_l, cin_l, hin_l, win_l, cout_l, hk_l, wk_l):
    RES = 0
    temp = 0
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' : 
            temp = hk_l[layer]*wk_l[layer]*cin_l[layer]*cout_l[layer]
        if layers_l[layer] == 'PW':
            temp = cin_l[layer]*cout_l[layer]
        if   layers_l[layer] == 'DW':
            temp = hk_l[layer]*wk_l[layer]*cin_l[layer]
        if layers_l[layer] == 'linear' :
            temp = cin_l[layer]*cout_l[layer]
        if layers_l[layer] == 'Sumnode':
            temp = cin_l[layer]*hin_l[layer]*win_l[layer]
        if temp > RES:
            RES = temp


    print(f"\nMax coefficients dim: {RES}\n")
    return RES


def max_layer_dim (layers_l, cin_l, hin_l, win_l, cout_l, hk_l, wk_l, data):
    RES = 0
    temp1 = 0 #input
    temp2 = 0 #wgt
    temp3 = 0 #output
    tot = 0
    max_layer =  0
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' : 
            temp2 = hk_l[layer]*wk_l[layer]*cin_l[layer]*cout_l[layer]
        if layers_l[layer] == 'PW':
            temp2 = cin_l[layer]*cout_l[layer]
        if   layers_l[layer] == 'DW':
            temp2 = hk_l[layer]*wk_l[layer]*cin_l[layer]
        if layers_l[layer] == 'linear' :
            temp2 = cin_l[layer]*cout_l[layer]
        if layers_l[layer] == 'Sumnode':
            temp2 = cin_l[layer]*hin_l[layer]*win_l[layer]

        temp1 = cin_l[layer]*hin_l[layer]*win_l[layer]

        if layer + 1 < len(layers_l): 
            temp3 = cin_l[layer + 1] * hin_l[layer + 1] * win_l[layer + 1] 
        else:
            temp3 = cout_l[layer] * hin_l[layer] * win_l[layer] 

        tot = temp1 + temp2 + temp3
        if tot > RES:
            RES = tot
            max_layer = layer

    multiplier = 2
    if data  == 'FP32':
        multiplier = 4
    print(f"Max Layer size : {multiplier*RES} bytes   @layer {max_layer}")
    return RES


"""
DNN Composer backend functions
"""

# Initializes the project folder with basic files
def InitProject(proj_folder_path):

    trainlib_src_folder = '../../lib/'
    proj_folder = proj_folder_path
    utils_folder = proj_folder + 'utils/'
    trainlib_dest_folder = proj_folder + 'lib/' 
    
    os.mkdir(proj_folder)
    os.mkdir(utils_folder)

    shutil.copy2('./utils/srcfiles/main.c', proj_folder)
    shutil.copy2('./utils/srcfiles/stats.h', proj_folder)
    shutil.copy2('./utils/srcfiles/dump_utils.py', utils_folder)
    shutil.copytree(trainlib_src_folder, trainlib_dest_folder)

    f = open(proj_folder+'readme.txt', 'w')
    f.write('To compile the application, run "make clean get_golden all run > log.txt".\nIf running on a board (not GVSoC), add "APP_CFLAGS += -DBOARD" to the user section of the Makefile (profiling of cycles only).\n')
    f.write('To modify the hyperparameters (learning rate, epochs, batch size still not implemented), \nedit the variables inside "utils/GM.py".\n')
    f.close()

    return





# Generates the Makefile
def GenerateMakefile(proj_folder_path, project_name, layers_l, NUM_CORES, data_type_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list):

    proj_folder = proj_folder_path
    makefile_name = proj_folder + 'Makefile'

    f = open(makefile_name, 'w')

    f.write('APP = ' + project_name + '\n\n')

    f.write('# User settings\n')
    f.write('NUM_CORES?=' + str(NUM_CORES) + '\n')
    f.write('#APP_CFLAGS += -DDEBUG' + '\n')
    f.write('#APP_CFLAGS += -DOPTIMIZE' + '     # Selects nth matmul to optimize execution\n')
    for layer in range(len(layers_l)):
        f.write('MATMUL_TYPE_FW_L'+str(layer)+'?='+str(opt_mm_fw_list[layer])+'         # Selects which optimized matmul to be used in FW (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
        f.write('MATMUL_TYPE_WG_L'+str(layer)+'?='+str(opt_mm_wg_list[layer])+'         # Selects which optimized matmul to be used in WEIGHT GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
        f.write('MATMUL_TYPE_IG_L'+str(layer)+'?='+str(opt_mm_ig_list[layer])+'         # Selects which optimized matmul to be used in IN GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
    f.write('# End of user settings\n\n')

    f.write('NUM_MATMULS?=24		# Available standard matmuls in the library' + '\n')
    f.write('TRAIN_LIB=./lib\n')
    f.write('TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources\n')
    f.write('APP_SRCS = main.c net.c\n\n')

    f.write('APP_CFLAGS += -I. -I$(TRAIN_LIB)/include\n')
    f.write('APP_CFLAGS += -O3 -g3\n')
    f.write('APP_CFLAGS += -DFABRIC\n')
    f.write('APP_CFLAGS += -DCLUSTER\n')
    f.write('APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)\n')
    f.write('APP_CFLAGS += -DPROF_NET\n')
    f.write('APP_CFLAGS += -mhwloopalign\n')
    for layer in range(len(layers_l)):
        f.write('APP_CFLAGS += -DMATMUL_TYPE_FW_L'+str(layer)+'=$'+str('{MATMUL_TYPE_FW_L')+str(layer)+str('}')+'\n')
        f.write('APP_CFLAGS += -DMATMUL_TYPE_WG_L'+str(layer)+'=$'+str('{MATMUL_TYPE_WG_L')+str(layer)+str('}')+'\n')
        f.write('APP_CFLAGS += -DMATMUL_TYPE_IG_L'+str(layer)+'=$'+str('{MATMUL_TYPE_IG_L')+str(layer)+str('}')+'\n')
    f.write('APP_LDFLAGS += -lm\n\n')

    f.write('# STATISTICS\n')
    f.write('APP_CFLAGS += -DSTATS\n\n')

    check_FP32 = False
    check_FP16 = False
    for layer in range(len(layers_l)):
        if data_type_l[layer] == 'FP32':
            check_FP32 = True
        elif data_type_l[layer] == 'FP16':
            check_FP16 = True

    f.write('# SOURCES\n')
    if check_FP32 == True:
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c\n\n')
    if check_FP16 == True:
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c\n\n')
    # if (check_FP16 and check_FP32) == False:
    #     print("[deployment_utils.GenerateMakefile] Data format not implemented!!\n")
    #     exit()

    f.write('# RULES\n')
    f.write('get_golden:\n')
    f.write('\tpython ./utils/GM.py\n')
    f.write('\n')

    f.write('include $(RULES_DIR)/pmsis_rules.mk\n')

    f.close()

    return


# Generates the Golden Model
def GenerateGM(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l, sumnode_connections):
    
    # Print DNN structure
    print("---------- DNN ARCHITECTURE ----------")
    for layer in range(len(layers_l)):
        h_out = math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer])
        w_out = math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer])
        print("Layer {}: {} {}, in=[{}, {}, {}], wgt=[{}, {}, {}, {}], out=[{}, {}, {}]".format(layer, data_type_l[layer], layers_l[layer], in_ch_l[layer], hin_l[layer], win_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], in_ch_l[layer], out_ch_l[layer], h_out, w_out))
    print("--------------------------------------")

    f = open(proj_folder_path+'utils/GM.py', 'w')

    f.write("import torch\n")
    f.write("import torch.nn as nn\n")
    f.write("import torch.optim as optim\n")
    f.write("import dump_utils as dump\n")
    f.write("import math\n")
    f.write("\n")

    # Define hyperparameters
    f.write("# Define hyperparameters\n")
    f.write("learning_rate = "+str(learning_rate)+"\n")
    f.write("batch_size = "+str(batch_size)+"\n")
    f.write("epochs = "+str(epochs)+"\n")
    f.write("\n")

    # Write sizes
    for layer in range(len(layers_l)):
        f.write("# LAYER "+str(layer)+" SIZES\n")
        f.write("l"+str(layer)+"_in_ch = "+str(in_ch_l[layer])+"\n")
        f.write("l"+str(layer)+"_out_ch = "+str(out_ch_l[layer])+"\n")
        f.write("l"+str(layer)+"_hk = "+str(hk_l[layer])+"\n")
        f.write("l"+str(layer)+"_wk = "+str(wk_l[layer])+"\n")
        f.write("l"+str(layer)+"_hin = "+str(hin_l[layer])+"\n")
        f.write("l"+str(layer)+"_win = "+str(win_l[layer])+"\n")
        # Padging and stride
        f.write("l"+str(layer)+"_hstr = "+str(h_str_l[layer])+"\n")
        f.write("l"+str(layer)+"_wstr = "+str(w_str_l[layer])+"\n")
        f.write("l"+str(layer)+"_hpad = "+str(h_pad_l[layer])+"\n")
        f.write("l"+str(layer)+"_wpad = "+str(w_pad_l[layer])+"\n")
    f.write("\n")

    # Write sizes to the header files 
    f.write("f = open('init-defines.h', 'w')\n")
    for layer in range(len(layers_l)):
        f.write("f.write('// Layer"+str(layer)+"\\n')\n")
        f.write("f.write('#define Tin_C_l"+str(layer)+" '+str(l"+str(layer)+"_in_ch)+'\\n')\n")
        f.write("f.write('#define Tout_C_l"+str(layer)+" '+str(l"+str(layer)+"_out_ch)+'\\n')\n")
        if layers_l[layer]  != 'Skipnode' and layers_l[layer]  != 'Sumnode':
            f.write("f.write('#define Tker_H_l"+str(layer)+" '+str(l"+str(layer)+"_hk)+'\\n')\n")
            f.write("f.write('#define Tker_W_l"+str(layer)+" '+str(l"+str(layer)+"_wk)+'\\n')\n")
        f.write("f.write('#define Tin_H_l"+str(layer)+" '+str(l"+str(layer)+"_hin)+'\\n')\n")
        f.write("f.write('#define Tin_W_l"+str(layer)+" '+str(l"+str(layer)+"_win)+'\\n')\n")
        f.write("f.write('#define Tout_H_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_hin-l"+str(layer)+"_hk+2*l"+str(layer)+"_hpad+l"+str(layer)+"_hstr)/l"+str(layer)+"_hstr))+'\\n')\n")
        f.write("f.write('#define Tout_W_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_win-l"+str(layer)+"_wk+2*l"+str(layer)+"_wpad+l"+str(layer)+"_wstr)/l"+str(layer)+"_wstr))+'\\n')\n")
        # Padding and stride
        if layers_l[layer]  != 'Skipnode' and layers_l[layer]  != 'Sumnode':
            f.write("f.write('#define Tstr_H_l"+str(layer)+" '+str(l"+str(layer)+"_hstr)+'\\n')\n")
            f.write("f.write('#define Tstr_W_l"+str(layer)+" '+str(l"+str(layer)+"_wstr)+'\\n')\n")
            f.write("f.write('#define Tpad_H_l"+str(layer)+" '+str(l"+str(layer)+"_hpad)+'\\n')\n")
            f.write("f.write('#define Tpad_W_l"+str(layer)+" '+str(l"+str(layer)+"_wpad)+'\\n')\n")
    f.write("f.close()\n\n")

    # Write hyperparameters to header
    f.write("f = open('init-defines.h', 'a')\n")
    f.write("f.write('\\n// HYPERPARAMETERS\\n')\n")
    f.write("f.write('#define LEARNING_RATE '+str(learning_rate)+'\\n')\n")
    f.write("f.write('#define EPOCHS '+str(epochs)+'\\n')\n")
    f.write("f.write('#define BATCH_SIZE '+str(batch_size)+'\\n')\n")
    f.write("f.close()\n\n")

    # Create input data and label
    f.write("\n# Simple input data \n")
    if (layers_l[0] == 'linear'):
        f.write("inp = torch.div(torch.ones(l0_in_ch), 100000)\n")
    elif (layers_l[0] == 'conv2d' or layers_l[0] == 'DW' or layers_l[0] == 'PW' or layers_l[0] == 'Skipnode'):
        f.write("inp = torch.torch.div(torch.ones(batch_size, l0_in_ch, l0_hin, l0_win), 1000)\n")
    # Throw error
    else:
        print("[deployment_utils.GenerateGM]: Input layer not valid!\n")
        exit()

    # Set input layer to half() in case of FP16
    if data_type_l[0] == 'FP16':
        f.write("inp = inp.half()\n")


    '''
    --------------------------------- MIXED PRECISION NON-WORKING - GENERATES A NON-WORKING DNN IN PYTORCH --------------------------------- 

    # Generate DNN model
    f.write("class DNN(nn.Module):\n")
    f.write("\tdef __init__(self):\n")
    f.write("\t\tsuper().__init__()\n")
    # Create neural network model
    for layer in range(len(layers_l)):
        # Layers
        if layers_l[layer] == "linear":
            f.write(Gtemp.linear_template(layer, in_ch_l[layer], out_ch_l[layer], "False", data_type_l[layer]))
        elif layers_l[layer] == "conv2d":
            f.write(Gtemp.conv2d_template(layer, in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False", data_type_l[layer]))
        elif layers_l[layer] == "DW":
            f.write(Gtemp.DW_template(layer, in_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False", data_type_l[layer]))
        elif layers_l[layer] == "PW":
            f.write(Gtemp.PW_template(layer, in_ch_l[layer], out_ch_l[layer], "False", data_type_l[layer]))
        # Activations
        elif layers_l[layer] == "ReLU":
            f.write(Gtemp.ReLU_template(layer, data_type_l[layer]))
        # Pooling
        elif layers_l[layer] == "MaxPool":
            f.write(Gtemp.MaxPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], data_type_l[layer]))
        elif layers_l[layer] == "AvgPool":
            f.write(Gtemp.AvgPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], data_type_l[layer]))
        # Throw error
        else:
            print("[deployment_utils.GenerateGM]: Layer {} not recognized!!\n".format(layer))
            exit()
    # Create Forward
    f.write("\n")
    f.write("\tdef forward(self, x):")
    for layer in range(len(layers_l)):
        # Vectorize inputs in case of linear layer
        if layers_l[layer] == 'linear':
            f.write("\n\t\tx = torch.reshape(x, (-1,))")
        # Set data format for each layer
        if layer == 0 and data_type_l[layer] == 'FP16':
            f.write("\n\t\tx = x.half()")
        elif data_type_l[layer] == 'FP32' and data_type_l[layer-1] != data_type_l[layer]:
            f.write("\n\t\tx = x.float()")
        elif data_type_l[layer] == 'FP16' and data_type_l[layer-1] != data_type_l[layer]:
            f.write("\n\t\tx = x.half()")
        # Forward layers 
        # (ReLU works with FP32 only)
        if layers_l[layer] == 'ReLU' and data_type_l[layer-1] == 'FP32' and data_type_l[layer] == 'FP16':
            f.write("\n\t\tx = self.l"+str(layer)+"(x.float()).half()")
        # Last layer
        elif layer == len(layers_l)-1:
            f.write("\n\t\tx = self.l"+str(layer)+"(x).float()")
        else:
            f.write("\n\t\tx = self.l"+str(layer)+"(x)")
    f.write("\n\t\treturn x\n")
    print("[deployment_utils.GenerateNet]: Setting last layer's output to float for PyTorch compatibility with loss function backward (future fix).")
    '''


    '''
    --------------------------------- WORKAROUND - FAKE FP16 FOR ALL DNN --------------------------------- 
    '''
    #Sumnode and Skipnode class generation 
    f.write("\nclass Sumnode():\n") 
    f.write("\tdef __init__(self, ls):\n") 
    f.write("\t\tself.MySkipNode = ls\n\n") 

    f.write("class Skipnode():\n") 
    f.write("\tdef __init__(self):\n") 
    f.write("\t\tself.data = 0\n\n") 
    f.write("\tdef __call__(self, x):\n") 
    f.write("\t\tself.data = x\n") 
    f.write("\t\treturn self.data\n\n")

    # Generate DNN model
    f.write("class DNN(nn.Module):\n")
    f.write("\tdef __init__(self):\n")
    f.write("\t\tsuper().__init__()\n")
    # Create neural network model
    for layer in range(len(layers_l)):
        # Layers
        if layers_l[layer] == "linear":
            f.write(Gtemp.linear_template(layer, in_ch_l[layer], out_ch_l[layer], "False", 'FP32'))
        elif layers_l[layer] == "conv2d":
            f.write(Gtemp.conv2d_template(layer, in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False", 'FP32'))
        elif layers_l[layer] == "DW":
            f.write(Gtemp.DW_template(layer, in_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False", 'FP32'))
        elif layers_l[layer] == "PW":
            f.write(Gtemp.PW_template(layer, in_ch_l[layer], out_ch_l[layer], "False", 'FP32'))
        # Activations
        elif layers_l[layer] == "ReLU":
            f.write(Gtemp.ReLU_template(layer, 'FP32'))
        # Pooling
        elif layers_l[layer] == "MaxPool":
            f.write(Gtemp.MaxPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], 'FP32'))
        elif layers_l[layer] == "AvgPool":
            f.write(Gtemp.AvgPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], 'FP32'))
        #Skipconn
        elif layers_l[layer] == "Skipnode": 
            f.write(Gtemp.Skipnode_template(layer)) 
        elif layers_l[layer] == "Sumnode": 
            f.write(Gtemp.Sumnode_template(layer, sumnode_connections[layer])) 
        # Throw error
        else:
            print("[deployment_utils.GenerateGM]: Layer {} not recognized!!\n".format(layer))
            exit()
    # Create Forward
    f.write("\n")
    f.write("\tdef forward(self, x):")
    for layer in range(len(layers_l)):

        variable = 'x'
        if sumnode_connections[layer] != -1:
            variable = f'y{sumnode_connections[layer]}' # Create a temporary variable for skip connections

        # Vectorize inputs in case of linear layer
        if layers_l[layer] == 'linear':
            f.write(f"\n\t\t{variable} = torch.reshape(x, (-1,))")
        # Set data format for each layer
        if layer == 0 and data_type_l[layer] == 'FP16':
            f.write(f"\n\t\tx = x.float()")
        elif data_type_l[layer] == 'FP32' and data_type_l[layer-1] != data_type_l[layer]:
            f.write(f"\n\t\tx = x.float()")
        elif data_type_l[layer] == 'FP16' and data_type_l[layer-1] != data_type_l[layer]:
            f.write(f"\n\t\tx = x.float()")
        # Forward layers 
        # (ReLU works with FP32 only)
        if layers_l[layer] == 'ReLU' and data_type_l[layer-1] == 'FP32' and data_type_l[layer] == 'FP16':
            f.write(f"\n\t\t{variable} = self.l"+str(layer)+f"({variable})")
        #Skipconn
        elif sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            f.write(f"\n\t\t{variable} = self.l{layer}(x)")
        elif layers_l[layer] == "Sumnode": 
            f.write(f"\n\t\tx = y{layer} + x\t# Sumnode") 
        # Last layer
        elif layer == len(layers_l)-1:
            f.write(f"\n\t\t{variable} = self.l"+str(layer)+"(x).float()")
        else:
            f.write("\n\t\tx = self.l"+str(layer)+"(x)")
        
    f.write("\n\t\treturn x\n")
    print("[deployment_utils.GenerateNet]: Setting last layer's output to float for PyTorch compatibility with loss function backward (future fix).")

    '''
    ---------------------------------   END OF WORKAROUND --------------------------------- 
    '''

    last_layer = len(layers_l) - 1

    # Initialize network
    f.write("\n# Initialize network\n")
    f.write("net = DNN()\n")
    f.write("for p in net.parameters():\n")
    f.write("\tnn.init.normal_(p, mean=0.0, std=1.0)\n")
    f.write("net.zero_grad()\n\n")

    # Write all-ones sample label
    f.write("\n# All-ones fake label \n")
    f.write("output_test = net(inp)\n")
    f.write("label = torch.ones_like(output_test)\n")

    # Write init weights to header file
    f.write("f = open('io_data.h', 'w')\n")
    f.write("f.write('// Init weights\\n')\n")
    for layer in range(len(layers_l)):
        if (layers_l[layer] != 'ReLU' and layers_l[layer] != 'MaxPool' and layers_l[layer] != 'AvgPool' and layers_l[layer] != 'Skipnode'  and layers_l[layer] != 'Sumnode'):
            f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
            if data_type_l[layer] == 'FP32':
                f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"] = {'+dump.tensor_to_string(net.l"+str(layer)+".weight.data)+'};\\n')\n")
            elif data_type_l[layer] == 'FP16':
                f.write("f.write('PI_L2 fp16 init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"] = {'+dump.tensor_to_string(net.l"+str(layer)+".weight.data)+'};\\n')\n")
            else:
                print("[deployment_utils.GenerateGM] Error in data type definition! (weight init)")
                exit()
        else:
            f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
            if data_type_l[layer] == 'FP32':
                f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"];\\n')\n")
            elif data_type_l[layer] == 'FP16':
                f.write("f.write('PI_L2 fp16 init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"];\\n')\n")
            else:
                print("[deployment_utils.GenerateGM] Error in data type definition! (weight init - empty ones)")
                exit()
    f.write("f.close()\n\n")

    # Define optimizer
    if optimizer == 'SGD':
        f.write("optimizer = optim."+str(optimizer)+"(net.parameters(), lr=learning_rate, momentum=0)\n")
    else:
        print("[deployment_utils.GenerateGM]: Invalid optimizer!!\n!")
        exit()
    f.write("loss_fn = nn."+str(loss_fn)+"()\n")
    f.write("\n")

    # Perform training
    f.write("# Train the DNN\n")
    f.write("for batch in range(epochs):\n")
    f.write("\toptimizer.zero_grad()\n")
    f.write("\tout = net(inp)\n")
    f.write("\tloss = loss_fn(out, label)\n")
    f.write("\tloss.backward()\n")
    f.write("\toptimizer.step()\n")
    
    # Inference after training
    f.write("\n# Inference once after training\n")
    f.write("out = net(inp)\n")
    f.write("\n")

    # Dump input and output of the network to the header file for the MCU
    f.write("f = open('io_data.h', 'a')\n")
    f.write("f.write('// Input and Output data\\n')\n")
    f.write("f.write('#define IN_SIZE "+str(in_ch_l[0]*win_l[0]*hin_l[0])+"\\n')\n")
    # Fake input data definition
    if data_type_l[0] == 'FP32':
        f.write("f.write('PI_L1 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\\n')\n")
    elif data_type_l[0] == 'FP16':
        f.write("f.write('PI_L1 fp16 INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\\n')\n")
    else:
        print("[deployment_utils.GenerateGM] Invalid input data size!")
    f.write("out_size = (int(math.floor(l"+str(last_layer)+"_hin-l"+str(last_layer)+"_hk+2*l"+str(last_layer)+"_hpad+l"+str(last_layer)+"_hstr)/l"+str(last_layer)+"_hstr)) * (int(math.floor(l"+str(last_layer)+"_win-l"+str(last_layer)+"_wk+2*l"+str(last_layer)+"_wpad+l"+str(last_layer)+"_wstr)/l"+str(last_layer)+"_wstr)) * l"+str(last_layer)+"_out_ch\n") 
    f.write("f.write('#define OUT_SIZE '+str(out_size)+'\\n')\n")
    # Fake output data and label definition
    if data_type_l[-1] == 'FP32':
        f.write("f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\\n')\n")
        f.write("f.write('PI_L2 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\\n')\n")
    elif data_type_l[-1] == 'FP16':
        f.write("f.write('PI_L2 fp16 REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\\n')\n")
        f.write("f.write('PI_L2 fp16 LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\\n')\n")    
    else:
        print("[deployment_utils.GenerateGM] Invalid output data size!")
    f.write("f.close()\n")

    f.close()

    return





# Generate the net.c and net.h files for the execution on PULP
def GenerateNet(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l, sumnode_connections, MAX_LAYER_DIM):


    data_type = data_type_l[0]
    data_size = 0
    suffix = ""
    if data_type == "FP32":
        data_size = 4
        suffix = ""
    else:
        data_size = 2
        suffix = "_fp16"

    # Generate net.h
    f = open(proj_folder_path+'net.h', 'w')

    f.write("// PULP Defines\n")
    f.write("#define STACK_SIZE      4096\n")
    f.write("\n")

    f.write("// Tolerance to check updated output\n")
    f.write("#define TOLERANCE 1e-12\n\n")

    f.write("// Training functions\n")
    f.write("void DNN_init();\n")
    f.write("void compute_loss();\n")
    f.write("void update_weights();\n")
    f.write("void forward();\n")
    f.write("void backward();\n")
    f.write("void net_step();\n")

    f.write("\n// Print and check functions\n")
    f.write("void print_output();\n")
    f.write("void check_post_training_output();\n")

    f.write("\n// DMA managment functions\n")
    f.write("void load_input(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void load_output(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void load_coeff(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void store_output(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void store_input(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void store_coeff(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void copy_struct_param(unsigned int from, unsigned int to, int size);\n")
    f.write("void get_input_dim(void * b);\n")
    f.write("void get_output_dim(void * b);\n")
    f.write("void get_weight_dim(void * b);\n")
    f.write("void reset_arguments();\n")
    f.write("void update_blob();\n")
    f.write("void reset_dim();\n")

    f.write(f"#define MAX_IN_SIZE {max_input_dim(layers_l, in_ch_l, hin_l, win_l)}\n")
    f.write(f"#define MAX_WGT_SIZE {max_wgt_dim(layers_l, in_ch_l, hin_l, win_l, out_ch_l, hk_l, wk_l)}\n")
    f.write(f"#define MAX_SIZE {2*MAX_LAYER_DIM}\n")
    f.close()    


    # Generate net.c
    f = open(proj_folder_path+'net.c', 'w')

    f.write("/**\n * INCLUDES\n**/\n\n")

    f.write("#include \"pulp_train.h\"\n")
    f.write("#include \"net.h\"\n")
    f.write("#include \"stats.h\"\n\n")
    f.write("#include \"init-defines.h\"\n")
    f.write("#include \"io_data.h\"\n")


    f.write("\n// Define structures and pointers to data in L1 memory\n")
    if data_type == 'FP32':
        f.write("PI_L1 float * IN_DATA , * IN_DIFF, * W_DATA, * W_DIFF, * OUT_DATA, * OUT_DIFF;\n")
        f.write("PI_L1 float BUFF[MAX_SIZE];\n")
        f.write("PI_L1 struct blob input_blob;\n")
        f.write("PI_L1 struct blob weight_blob;\n")
        f.write("PI_L1 struct blob output_blob;\n")
        f.write("PI_L1 struct blob temp_blob;\n")
        f.write("PI_L1 struct Linear_args linear_args;\n")
        f.write("PI_L1 struct Conv2D_args conv2d_args;\n")
        f.write("PI_L1 struct PointWise_Conv_args PW_args;\n")
        f.write("PI_L1 struct DepthWise_Conv_args DW_args;\n")
        f.write("PI_L1 struct act_args act_args;\n")
        f.write("PI_L1 struct SkipConn_args resconn_args;\n")
        f.write("PI_L1 float * t;\n")
    elif data_type == 'FP16':
        f.write("PI_L1 fp16 * IN_DATA , * IN_DIFF, * W_DATA, * W_DIFF, * OUT_DATA, * OUT_DIFF;\n")
        f.write("PI_L1 fp16 BUFF[MAX_SIZE];\n")
        f.write("PI_L1 struct blob_fp16 input_blob;\n")
        f.write("PI_L1 struct blob_fp16 weight_blob;\n")
        f.write("PI_L1 struct blob_fp16 output_blob;\n")
        f.write("PI_L1 struct blob_fp16 temp_blob;\n")
        f.write("PI_L1 struct Linear_args_fp16 linear_args;\n")
        f.write("PI_L1 struct Conv2D_args_fp16 conv2d_args;\n")
        f.write("PI_L1 struct PointWise_Conv_args_fp16 PW_args;\n")
        f.write("PI_L1 struct DepthWise_Conv_args_fp16 DW_args;\n")
        f.write("PI_L1 struct act_args_fp16 act_args;\n")
        f.write("PI_L1 struct SkipConn_args_fp16 resconn_args;\n")
        f.write("PI_L1 fp16 * t;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid last layer data type!")
        exit()
    
    f.write("PI_L1 pi_cl_dma_cmd_t * cmd_store;\n")
    f.write("PI_L1 pi_cl_dma_cmd_t * cmd_load;\n")

    f.write("\n\n\n/**\n * DATA\n**/\n")

    f.write("\n// Define loss\n")
    if data_type_l[-1] == 'FP32':
        f.write("PI_L1 float loss = 0;\n")
    elif data_type_l[-1] == 'FP16':
        f.write("PI_L1 fp16 loss = 0;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid last layer data type!")
        exit()


    f.write("\n// Define DNN blobs\n")
    for layer in range(len(layers_l)):
        if data_type_l[layer] == 'FP32':
            f.write("PI_L2 struct blob layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, layer"+str(layer)+"_out;\n")
        elif data_type_l[layer] == 'FP16':
            f.write("PI_L2 struct blob_fp16 layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, layer"+str(layer)+"_out;\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for blob definition @Layer{}!".format(layer))
            exit()


    f.write("\n// Define DNN layer structures\n")
    f.write("PI_L1 struct vect_sum_args vect_sum_args;\n")
    f.write("PI_L1 struct vect_sum_args_fp16 vect_sum_args_fp16;\n")
    for layer in range(len(layers_l)):
        # Define FP32 structure
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'linear':
                f.write("PI_L2 struct Linear_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L2 struct Conv2D_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L2 struct PointWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L2 struct DepthWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L2 struct act_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass 
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L2 struct SkipConn_args l"+str(layer)+"_args;\n")
            else:
                print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")
        # Define FP16 structure
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'linear':
                f.write("PI_L2 struct Linear_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L2 struct Conv2D_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L2 struct PointWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L2 struct DepthWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L2 struct act_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L2 struct SkipConn_args_fp16 l"+str(layer)+"_args;\n")
            else:
                print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")
        # Invalid data type
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for structure initialization @Layer{}!".format(layer))


    pooling_exist = False
    for layer in range(len(layers_l)):
        if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
            pooling_exist = True
    if pooling_exist:
        f.write("\n// Define Pooling Structures\n")
        for layer in range(len(layers_l)):
            if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
                if data_type_l[layer] == 'FP32':
                    f.write("PI_L2 struct pool_args l"+str(layer)+"_pool_args;\n")
                elif data_type_l[layer] == 'FP16':
                    f.write("PI_L2 struct pool_args_fp16 l"+str(layer)+"_pool_args;\n")
                else:
                    print("[deployment_utils.GenerateNet] Invalid data type for pooling initialization @Layer{}!".format(layer))
                    exit()


    f.write("\n// Define kernel tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 float l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            else:    
                f.write("PI_L2 float l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 fp16 l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            else:    
                f.write("PI_L2 fp16 l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for kernel definition @Layer{}!".format(layer))
            exit()

    f.write("\n// Define kernel grad tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 float l"+str(layer)+"_ker_diff[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                pass
            else:    
                f.write("PI_L2 float l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 fp16 l"+str(layer)+"_ker_diff[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                pass
            else:    
                f.write("PI_L2 fp16 l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for kernel grad definition @Layer{}!".format(layer))
            exit()

    f.write("\n// Define I/O tensors\n")

    previous_was_skip = False 
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if not previous_was_skip: # If the previous layer was a Skipnode, then do not generate layer in and diff
            if data_type_l[layer] == 'FP32':
                f.write("PI_L2 float l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L2 float l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Define FP16 tensors
            elif data_type_l[layer] == 'FP16':
                f.write("PI_L2 fp16 l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L2 fp16 l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Data type error
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for I/O definition @Layer{}!".format(layer))
                exit()

        if layers_l[layer] == 'Skipnode':
            previous_was_skip = True
        else:
            previous_was_skip = False
    # Write IM2COL buffers
    im2col_flag = False
    im2col_type = 'FW'  # 'FW' or 'BW'
    im2col_max_memocc = 0
    im2col_layer_index = 0
    im2col_byte_length = 0
    im2col_max_data_type = 'FP32'
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d': # or layers_l[layer] == 'DW':
            if data_type_l[layer] == 'FP32':
                im2col_byte_length = 4
            elif data_type_l[layer] == 'FP16':
                im2col_byte_length = 2
            im2col_flag = True
            i2c_mem = 0
            i2c_FW = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer]) * math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer]) * im2col_byte_length
            i2c_BW = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * hin_l[layer] * win_l[layer] * im2col_byte_length
            if i2c_FW > i2c_BW:
                i2c_mem = i2c_FW
                im2col_type = 'FW'
            else:
                i2c_mem = i2c_BW
                im2col_type = 'BW'
            if i2c_mem > im2col_max_memocc:
                im2col_max_memocc = i2c_mem
                im2col_layer_index = layer
                im2col_max_data_type = data_type_l[layer]
    if im2col_flag == True:
        if im2col_type == 'FW':
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            if im2col_max_data_type == 'FP32':
                f.write("PI_L1 float im2col_buffer[Tin_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tout_H_l"+str(im2col_layer_index)+"*Tout_W_l"+str(im2col_layer_index)+"];\n")
            elif im2col_max_data_type == 'FP16':
                f.write("PI_L1 fp16 im2col_buffer[Tin_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tout_H_l"+str(im2col_layer_index)+"*Tout_W_l"+str(im2col_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for im2col!!")
                exit()
        else:
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            if im2col_max_data_type == 'FP32':
                f.write("PI_L1 float im2col_buffer[Tout_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tin_H_l"+str(im2col_layer_index)+"*Tin_W_l"+str(im2col_layer_index)+"];\n")
            elif im2col_max_data_type == 'FP16':
                f.write("PI_L1 fp16 im2col_buffer[Tout_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tin_H_l"+str(im2col_layer_index)+"*Tin_W_l"+str(im2col_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for im2col!!")
                exit()

    # Write in grad transposition / blocktranspose buffer
    bt_flag = False
    bt_max_memocc = 0
    bt_layer_index = 0
    wgt_grad_pw = False
    bt_max_data_type = 'FP32'
    for layer in range(len(layers_l)):
        # Check layer data layout
        data_layout = 'CHW'     # Change to input list of data layouts
        if (layers_l[layer] == 'conv2d' or layers_l[layer] == 'PW') and layer == 0:
            bt_flag = True
            bt_layer_index = 0
        elif (layers_l[layer] == 'conv2d' or layers_l[layer] == 'PW') and layer > 0:
            bt_flag = True
            bt_mem = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * out_ch_l[layer]
            if bt_mem > bt_max_memocc:
                bt_max_memocc = bt_mem
                bt_layer_index = layer
                bt_max_data_type = data_type_l[layer]
        # Special conditions in case of HWC
        if (data_layout == 'HWC' and layers_l[layer] == 'PW'):
            # Special allocation for weight grad in HWC
            bt_flag = True
            bt_mem = in_ch_l[layer] * hin_l[layer] * win_l[layer]
            if data_type_l[layer] == 'FP16':
                hout = hin_l[layer]; wout = win_l[layer]
                bt_mem += out_ch_l[layer] * hout * wout
            if bt_mem > bt_max_memocc:
                bt_max_memocc = bt_mem
                bt_layer_index = layer
                bt_max_data_type = data_type_l[layer]
                wgt_grad_pw = True
    if (bt_flag == True) and (wgt_grad_pw == False):
        f.write("\n// Define transposition / block transposition buffer for all conv2d and PW layers\n")
        if bt_layer_index == 0:
            f.write("PI_L1 float bt_buffer[1];")
        elif bt_layer_index > 0:
            if bt_max_data_type == 'FP32':
                f.write("PI_L1 float bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tout_C_l"+str(bt_layer_index)+"*Tker_H_l"+str(bt_layer_index)+"*Tker_W_l"+str(bt_layer_index)+"];\n")
            elif bt_max_data_type == 'FP16':
                f.write("PI_L1 fp16 bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tout_C_l"+str(bt_layer_index)+"*Tker_H_l"+str(bt_layer_index)+"*Tker_W_l"+str(bt_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for blocktranspose!")
                exit()
    elif (bt_flag == True) and (wgt_grad_pw == True):
        f.write("\n// Define transposition / block transposition buffer for all conv2d and PW layers\n")
        if bt_max_data_type == 'FP32':
            f.write("PI_L1 float bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tin_H_l"+str(bt_layer_index)+"*Tin_W_l"+str(bt_layer_index)+"];\n")
        elif bt_max_data_type == 'FP16':
            f.write("PI_L1 fp16 bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tin_H_l"+str(bt_layer_index)+"*Tin_W_l"+str(bt_layer_index)+"+Tout_C_l"+str(bt_layer_index)+"*Tout_H_l"+str(bt_layer_index)+"*Tout_W_l"+str(bt_layer_index)+"];\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for pw transp buffer definition!\n")
            exit()


    # Define tensors to backpropagate the output error
    f.write("\n// Define error propagation tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layer > 0:
                f.write("PI_L2 float l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
            if (layer == len(layers_l)-1):
                f.write("PI_L2 float l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layer > 0:
                f.write("PI_L2 fp16 l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
            if (layer == len(layers_l)-1):
                f.write("PI_L2 fp16 l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for input grad definition @Layer{}!".format(layer))
            exit()  
        
      

    # Define buffer for mixed precision propagation
    previous_type = data_type_l[0]
    is_mixed_precision = False
    curr_cast_in_size = 0
    curr_cast_out_size = 0
    curr_max_size = 0
    is_max_input = False
    max_cast_buffer_index = 0
    max_cast_buffer_size = 0
    max_cast_buffer_type = 'FP32'
    for layer in range(len(layers_l)):
        # Output size for current layer
        h_out = math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer])
        w_out = math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer])
        # Find if there are mixed types
        if data_type_l[layer] != previous_type:
            is_mixed_precision = True
            # Find biggest size
            curr_cast_in_size = in_ch_l[layer] * hin_l[layer] * win_l[layer]
            curr_cast_out_size = out_ch_l[layer] * h_out * w_out
            if curr_cast_in_size > curr_cast_out_size:
                curr_max_size = curr_cast_in_size
                is_max_input = True
            else:
                curr_max_size = curr_cast_out_size
                is_max_input = False
            if curr_max_size > max_cast_buffer_size:
                max_cast_buffer_size = curr_max_size
                max_cast_buffer_type = data_type_l[layer-1]
                max_cast_buffer_index = layer
        previous_type = data_type_l[layer]

    # Allocate buffer
    if is_mixed_precision:
        f.write("\n// Define cast buffer to manage mixed precision (size="+str(max_cast_buffer_size)+")\n")
        if max_cast_buffer_type == 'FP32':
            if is_max_input:
                f.write("PI_L1 float cast_buffer[Tin_C_l"+str(max_cast_buffer_index)+" * Tin_H_l"+str(max_cast_buffer_index)+" * Tin_W_l"+str(max_cast_buffer_index)+"];\n")
            else:
                f.write("PI_L1 float cast_buffer[Tout_C_l"+str(max_cast_buffer_index)+" * Tout_H_l"+str(max_cast_buffer_index)+" * Tout_W_l"+str(max_cast_buffer_index)+"];\n")
        elif max_cast_buffer_type == 'FP16':
            if is_max_input:
                f.write("PI_L1 fp16 cast_buffer[Tin_C_l"+str(max_cast_buffer_index)+" * Tin_H_l"+str(max_cast_buffer_index)+" * Tin_W_l"+str(max_cast_buffer_index)+"];\n")
            else:
                f.write("PI_L1 fp16 cast_buffer[Tout_C_l"+str(max_cast_buffer_index)+" * Tout_H_l"+str(max_cast_buffer_index)+" * Tout_W_l"+str(max_cast_buffer_index)+"];\n")
        else:
            print("[deployment_utils.GenerateNet]: Invalid data type for mixed precision buffer!")
            exit() 



    f.write("\n// Loss function configuration structure\n")
    if data_type_l[-1] == 'FP32':
        f.write("PI_L1 struct loss_args loss_args;\n")
    elif data_type_l[-1] == 'FP16':
        f.write("PI_L1 struct loss_args_fp16 loss_args;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid data type for loss definition!")
        exit()
        

    f.write("\n\n\n/**\n * DNN BACKEND FUNCTIONS\n**/\n")

    f.write("\n// DNN initialization function\n")
    f.write("void DNN_init()\n{\n")
    f.write("\n// Assign pointers in L1\n")
    f.write("IN_DATA = BUFF;\n")
    f.write("IN_DIFF = BUFF;\n")
    f.write("W_DATA = BUFF;\n")
    f.write("W_DIFF = BUFF;\n")
    f.write("OUT_DATA = BUFF;\n")
    f.write("OUT_DIFF = BUFF;\n")
    f.write("update_blob();\n")
    f.write("reset_arguments();\n\n")
    for layer in range(len(layers_l)):
        if layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)\t\t\tl0_in[i] = INPUT[i];\n")
            if layers_l[layer] != 'Skipnode' and layers_l[layer] != 'Sumnode':
                f.write("  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)\t\tl0_ker[i] = init_WGT_l0[i];\n")
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            if layers_l[layer] == 'DW':
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
            elif layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool':
                f.write("  //   Pooling kernel (no parameters)\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                f.write("  //   Resconn layer (no parameters)\n")
            else:
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        elif layer == len(layers_l)-1:
            if layers_l[layer] != 'Skipnode' and layers_l[layer] != 'Sumnode':
                f.write("  // Layer "+str(layer)+"\n")
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

    # Mixed precision check
    C_data_type = 'float'
    f.write("\n  // Connect tensors to blobs\n")
    previous_was_skip = 0
    
    for layer in range(len(layers_l)):
        
        # Find data type for each layer
        if data_type_l[layer] == 'FP32':
            C_data_type = 'float'
        elif data_type_l[layer] == 'FP16':
            C_data_type = 'fp16'
        else:
            print("[deployment_utils.GenerateNet]: Invalid data type for structure assignment @layer{}!".format(layer))
            exit()
        f.write(f"\n\n//Connecting {layers_l[layer]}\n")
        # Verify how many Skipnodes comes after current layer (for diff connections)
        lookahead = 0
        if (layer + 1) <  len(layers_l):
            for l in range(len(layers_l) - layer - 1):
                if sumnode_connections[layer + l + 1] < 0 or layers_l[layer + l + 1] == 'Sumnode':
                    break
                else:
                    lookahead += 1
        # DNN is 1 layer long
        if len(layers_l) == 1:
            f.write("  layer"+str(layer)+"_in.data = l0_in;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l0;\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.data = l0_ker;\n")
            f.write("  layer"+str(layer)+"_wgt.diff = l0_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tker_H_l0*Tker_W_l0;\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.data = l0_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l0_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l0;\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l0;\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l0;\n")
        # First layer connection
        elif layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            if layers_l[0] != 'Skipnode': # Avoid weight assignment for Skip Connections
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if sumnode_connections[layer] < 0 or layers_l[layer] == 'Sumnode':
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(layer + 1 + lookahead)+"_in_diff;\n")   
                    else:
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")
                # End of assignment       
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
            else:
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")             
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        # Hidden layers
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip)+"_in;\n")
            if layers_l[layer] != 'Skipnode':
                if (layer - previous_was_skip) > 0: # Avoid assignement of l0_in_diff
                    f.write("  layer"+str(layer)+"_in.diff = l"+str(layer)+"_in_diff;\n")
            else:
                f.write(f"\tlayer{layer}_in.diff = l{sumnode_connections[layer]}_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            if layers_l[layer] != 'Skipnode':   # Avoid weight assignment for Skipnodes and out data assignement
                if layers_l[layer]  != 'Sumnode':    # Different weight assignement for Sumnodes
                    f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                    if layers_l[layer] == 'DW':
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    else:
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.data = layer"+str(sumnode_connections[layer])+"_out.data;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = layer"+str(sumnode_connections[layer])+"_out.diff;\n")
                    f.write("  layer"+str(layer)+"_wgt.C = layer"+str(sumnode_connections[layer])+"_out.C;\n")
                    f.write("  layer"+str(layer)+"_wgt.H = layer"+str(sumnode_connections[layer])+"_out.H;\n")
                    f.write("  layer"+str(layer)+"_wgt.W = layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    f.write("  layer"+str(layer)+"_wgt.dim = layer"+str(sumnode_connections[layer])+"_out.C*layer"+str(sumnode_connections[layer])+"_out.H*layer"+str(sumnode_connections[layer])+"_out.W;\n")
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if sumnode_connections[layer] == -1 or layers_l[layer] == 'Sumnode':
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1+lookahead)+"_in_diff;\n")
                    else:     
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")
                # End of assignment     
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
            else:
                f.write(f"\tlayer{layer}_out = layer{layer}_in;\n")
        # Last layer
        elif layer == len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.diff = l"+str(layer + lookahead)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            if layers_l[layer] !=  'Sumnode':
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
            else:
                    f.write("  layer"+str(layer)+"_wgt.data = layer"+str(sumnode_connections[layer])+"_out.data;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = layer"+str(sumnode_connections[layer])+"_out.diff;\n")
                    f.write("  layer"+str(layer)+"_wgt.C = layer"+str(sumnode_connections[layer])+"_out.C;\n")
                    f.write("  layer"+str(layer)+"_wgt.H = layer"+str(sumnode_connections[layer])+"_out.H;\n")
                    f.write("  layer"+str(layer)+"_wgt.W = layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    f.write("  layer"+str(layer)+"_wgt.dim = layer"+str(sumnode_connections[layer])+"_out.C*layer"+str(sumnode_connections[layer])+"_out.H*layer"+str(sumnode_connections[layer])+"_out.W;\n")
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer)+"_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

        if sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            previous_was_skip += 1
           
        else:
            previous_was_skip = 0
            

    f.write("\n  // Configure layer structures\n")
    first_is_skip = False # Avoid calculation of gradient if the first Layer is a skipnode
    if sumnode_connections[0] != -1:
        first_is_skip = True
    previous_was_skip = 0
    for layer in range(len(layers_l)):
        f.write("  // Layer "+str(layer)+"\n")
        if layer == 0:
            skip_inputgrad = 1
        elif layer - previous_was_skip <= 0: # If the 0 layer is a Skipnode, then layer1's diff is the input gradient
            skip_inputgrad = 1
        else: 
            skip_inputgrad = 0
        # Write configuration templates
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_config_template(layer, skip_inputgrad, data_type_l[layer]))
        elif layers_l[layer] == 'conv2d':
            f.write(ntemp.conv2d_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer]))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_config_template(layer, skip_inputgrad, data_type_l[layer]))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer]))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_config_template(layer, data_type_l[layer]))
        elif layers_l[layer] == 'MaxPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'AvgPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'Sumnode':
            f.write(ntemp.resconn_config_template(layer, sumnode_connections[layer], first_is_skip))
            first_is_skip = False
        elif layers_l[layer] == 'Skipnode':
            pass
        else:
            print("[deployment_utils.GenerateNet] Undefined layer "+str(layer)+" (unable to write configuration structure)!!")
        if sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            previous_was_skip += 1
        else:
            previous_was_skip = 0

    pooling_exist = False
    for layer in range(len(layers_l)):
        if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
            pooling_exist = True
    if pooling_exist:
        f.write("\n  // Connect blobs to pooling structures\n")
        for layer in range(len(layers_l)):
            if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
                f.write("  // Layer "+str(layer)+"\n")
                f.write("  l"+str(layer)+"_pool_args.input = &layer"+str(layer)+"_in;\n")
                f.write("  l"+str(layer)+"_pool_args.output = &layer"+str(layer)+"_out;\n")
                f.write("  l"+str(layer)+"_pool_args.Hker = Tker_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Wker = Tker_W_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Hstride = Tstr_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Wstride = Tstr_W_l"+str(layer)+";\n")
    f.write("}\n\n")


    f.write("\n// Forward pass function\n")
    f.write("void forward()\n{\n")
    f.write("\treset_dim();\n")
    f.write("\tload_input(&layer0_in, 1);\n")
    previous_was_skip = False
    for layer in range(len(layers_l)):
        if layer > 0:
            f.write("\treset_dim();\n")
            f.write(f"\tload_input(&layer{layer}_in, 1);\n")

        if layers_l[layer] != 'Skipnode' and layers_l[layer] != 'ReLU':
            f.write(f"\tload_coeff(&layer{layer}_wgt, 1);\n")
            if layers_l[layer] != 'Sumnode':
                f.write(f"\tcopy_struct_param((unsigned int) &l{layer}_args, (unsigned int) &{layers_l[layer]}_args, sizeof({layers_l[layer]}_args));\n")
        f.write(f"\tget_output_dim(&layer{layer}_out);\n")
        # Generate layer template
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'conv2d':
            f.write(ntemp.conv2d_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'AvgPool':
            f.write(ntemp.AvgPool_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'MaxPool':
            f.write(ntemp.MaxPool_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'Skipnode':
            pass
        elif layers_l[layer] == 'Sumnode':
            f.write(ntemp.residualconn_template_FW(layer, data_type_l[layer]))
        else:
            print("[deployment_utils.GenerateNet]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
        if layers_l[layer] != 'Skipnode':
            f.write(f"\tstore_output(&layer{layer}_out, 1);\n\n")
        else:
            f.write(f"\tstore_input(&layer{layer}_out, 1);\n\n")
        # Insert casting operator for data type variation
        if layer < len(layers_l)-1 and data_type_l[layer] != data_type_l[layer+1]:
            if data_type_l[layer] == 'FP32' and data_type_l[layer+1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(layer, "FW", data_type_l[layer]))
            elif data_type_l[layer] == 'FP16' and data_type_l[layer+1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(layer, "FW", data_type_l[layer]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[layer], data_type_l[layer+1], layer))

        # Check if current layer is Skipnode
        if sumnode_connections[layer] < 0 or layers_l[layer] == 'Sumnode':
            previous_was_skip = False
        else:
            previous_was_skip = True
    f.write("}\n")


    f.write("\n// Backward pass function\n")
    f.write("void backward()\n{\n")
    for layer in range(len(layers_l)):
        lay = len(layers_l) - layer - 1
        # Generate backward layer template
        is_skipderivation = False # Bool for Skipnode and layer after Skipnodes detection
        if layers_l[lay] != 'Sumnode' and sumnode_connections[lay] > -1:
            is_skipderivation = True

        skip_in_grad = 0
        if lay == 0:
            skip_in_grad = 1

        target_layer = lay
        if is_skipderivation: # Check for target layer's input for diff calculation of Skipnode derivations
            for l in range(len(layers_l)):
                if sumnode_connections[lay + l ] < 0:
                    break
                else:
                    target_layer += 1

        
        f.write("\n\treset_dim();\n")

        if layers_l[lay] != 'Sumnode':
            if layers_l[lay] == 'Skipnode':
                f.write(f"\tload_input(&layer{target_layer}_in, 0);\n")
            else:
                f.write(f"\tload_input(&layer{target_layer}_in, 1);\n")

        if layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and layers_l[lay] != 'ReLU':
            f.write(f"\tload_coeff(&layer{lay}_wgt, 1);\n")

        f.write(f"\tload_output(&layer{lay}_out, 2);\n")

        # Copy struct info 
        if layers_l[lay] != 'Skipnode' and layers_l[lay] != 'Sumnode' and layers_l[lay] != 'ReLU':
            f.write(f"\tcopy_struct_param((unsigned int) &l{lay}_args, (unsigned int) &{layers_l[lay]}_args, sizeof(l{lay}_args));\n")

        if layers_l[lay] == 'linear':
            f.write(ntemp.linear_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'conv2d':
            f.write(ntemp.conv2d_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'DW':
            f.write(ntemp.DW_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'PW':
            f.write(ntemp.PW_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'ReLU':
            f.write(ntemp.ReLU_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'AvgPool':
            f.write(ntemp.AvgPool_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'MaxPool':
            f.write(ntemp.MaxPool_template_BW(lay, data_type_l[lay]))
        elif layers_l[lay] == 'Skipnode':
            f.write(ntemp.residualconn_template_sum_BW(sumnode_connections[lay], data_type_l[lay], target_layer))
        elif layers_l[lay] == 'Sumnode':
            #f.write(ntemp.residualconn_template_copy_BW(lay, data_type_l[lay]))
            f.write(f"\tstore_output(&layer{lay}_in, 0);\n")
        else:
            print("[deployment_utils.GenerateNet]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
        # Insert casting operator for data type variation
        if lay < len(layers_l)-1 and lay > 0 and data_type_l[lay] != data_type_l[lay-1]:
            if data_type_l[lay] == 'FP32' and data_type_l[lay-1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(lay, "BW", data_type_l[lay]))
            elif data_type_l[lay] == 'FP16' and data_type_l[lay-1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(lay, "BW", data_type_l[lay]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[lay], data_type_l[lay-1], lay))



        if sumnode_connections[lay] != -1 and layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and skip_in_grad==0:
            f.write(f"\tload_output(&layer{target_layer}_in, 0);\n")
            f.write(ntemp.sum(lay, data_type_l[lay]))
        

        if layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and layers_l[lay] != 'ReLU':
            f.write(f"\tstore_coeff(&layer{lay}_wgt, 0);\n")

        if lay > 0 and layers_l[lay] != 'Sumnode':
            f.write(f"\tstore_input(&layer{target_layer}_in, 0);\n")
    f.write("}\n")


    f.write("\n// Compute loss and output gradient\n")
    f.write("void compute_loss()\n{\n")

    if loss_fn == "MSELoss":
        float_size = 2
        if data_type_l[0] == 'FP32':
            float_size = 4
        f.write("  loss_args.output = &output_blob;\n")
        f.write("  loss_args.target = output_blob.diff;\n")
        f.write("  loss_args.wr_loss = &loss;\n")
        f.write(f"  pi_cl_dma_cmd((uint32_t) (LABEL), (uint32_t) (output_blob.diff), {float_size}*OUT_SIZE, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
        f.write("  pi_cl_dma_cmd_wait(cmd_load);\n")

        if data_type_l[-1] == 'FP32':
            f.write("  pulp_MSELoss(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_MSELoss_fp16(&loss_args);\n")
        else:
            print("[deplyment_utils.GenerateNet]: Invalid loss type!")
            exit()
        f.write(f"  store_output(&layer{len(layers_l)-1}_out, 2);\n")
    else:
        print("[deployment_utils.GenerateNet]: Loss function not valid for PULP deployment!!")
        exit()

    f.write("}\n")


    f.write("\n// Function to update the network\n")
    f.write("void update_weights()\n{\n")

    for layer in range(len(layers_l)):
        if layers_l[layer] == 'linear' or layers_l[layer] == 'conv2d' or layers_l[layer] == 'DW' or layers_l[layer] == 'PW':
            if data_type_l[layer] == 'FP32':
                f.write("  struct optim_args opt_l"+str(layer)+";\n")
            elif data_type_l[layer] == 'FP16':
                f.write("  struct optim_args_fp16 opt_l"+str(layer)+";\n")
            else:
                print("[deployment_utils.GenerateNet]: Invalid data type for optimizer structure generation @layer{}!".format(layer))  
            f.write("  opt_l"+str(layer)+".weights = &weight_blob;\n")
            f.write("  opt_l"+str(layer)+".learning_rate = LEARNING_RATE;\n")
            f.write(f"  load_coeff(&layer{layer}_wgt, 2);\n")
            if optimizer == "SGD":
                if data_type_l[layer] == 'FP32':
                    f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l"+str(layer)+");\n")
                elif data_type_l[layer] == 'FP16':
                    f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp16, &opt_l"+str(layer)+");\n")
                else:
                    print("[deployment_utils.GenerateNet]: Invalid data type for gradient descent @Layer{}!".format(layer))
            else:
                print("[deployment_utils.GenerateNet]: Invalid optimizer for PULP deployment!!")
                exit()
            f.write(f"  store_coeff(&layer{layer}_wgt, 2);\n\n")
    f.write("}\n")


    f.write("\n\n\n/**\n * DATA VISUALIZATION AND CHECK TOOLS\n**/\n")

    f.write("\n// Function to print FW output\n")
    f.write("void print_output()\n{\n")
    output_index = len(layers_l) - 1
    f.write("  printf(\"\\nLayer "+str(output_index)+" output:\\n\");\n\n")
    f.write("  for (int i=0; i<Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+"; i++)\n  {\n")
    f.write("    printf(\"%f \", l"+str(output_index)+"_out[i]);\n")
    f.write("    // Newline when an output row ends\n")
    f.write("    // if(!(i%Tout_W_l"+str(output_index)+")) printf(\"\\n\");\n")
    f.write("    // Newline when an output channel ends\n")
    f.write("    if(!(i%Tout_W_l"+str(output_index)+"*Tout_H_l"+str(output_index)+")) printf(\"\\n\");\n")
    f.write("  }\n")
    f.write("}\n")


    f.write("\n// Function to check post-training output wrt Golden Model (GM)\n")
    f.write("void check_post_training_output()\n{\n")

    output_index = len(layers_l) - 1
    f.write("  int integrity_check = 0;\n")
    if data_type_l[output_index] == 'FP32':
        f.write("  integrity_check = verify_tensor(l"+str(output_index)+"_out, REFERENCE_OUTPUT, Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+", TOLERANCE);\n")
    elif data_type_l[output_index] == 'FP16':
        f.write("  integrity_check = verify_tensor_fp16(l"+str(output_index)+"_out, REFERENCE_OUTPUT, Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+", TOLERANCE);\n")
    else:
        print("[deployment_utils.GenerateNet]: Invalid inference verification data type!!")
        exit()
    f.write("  if (integrity_check > 0)\n")
    f.write("    printf(\"\\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\\n\");\n")

    f.write("}\n")



    f.write("\n\n\n/**\n * DNN MODEL TRAINING\n**/\n")

    f.write("\n// Call for a complete training step\n")
    f.write("void net_step()\n{\n")

    f.write("  printf(\"Initializing network..\\n\");\n")
    f.write("  DNN_init();\n")

    f.write("  printf(\"Testing DNN initialization forward..\");\n")
    f.write("  forward();\n")
    f.write("  print_output();\n\n")

    f.write("  #ifdef PROF_NET\n")
    f.write("  INIT_STATS();\n  PRE_START_STATS();\n  START_STATS();\n")
    f.write("  #endif\n\n")

    f.write("  for (int epoch=0; epoch<EPOCHS; epoch++)\n  {\n")
    f.write("    forward();\n")
    f.write("    compute_loss();\n")
    f.write("    backward();\n")
    f.write("    update_weights();\n")
    f.write("  }\n\n")

    f.write("  #ifdef PROF_NET\n")
    f.write("  STOP_STATS();\n")
    f.write("  #endif\n\n")

    f.write("  // Check and print updated output\n")
    f.write("  forward();\n")
    f.write("  printf(\"Checking updated output..\\n\");\n")
    f.write("  check_post_training_output();\n")
    f.write("  print_output();\n")

    f.write("}\n")

    data_size = 0
    suffix = ""

    if data_type_l[0] == 'FP32':
        data_size = 4
        suffix = ""
    else :
        data_size = 2
        suffix = "_fp16"

    f.write("\n// Functions for DMA managment\n")
    f.write("\nvoid load_coeff(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_weight_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")

    f.write("\nvoid load_input(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_input_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")

    f.write("\nvoid load_output(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_output_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")

    f.write("\nvoid store_output(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n") 

    f.write("\nvoid store_coeff(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n")

    f.write("\nvoid store_input(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n")

    f.write("\nvoid get_input_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\tinput_blob.C = src->C;\n")
    f.write("\tinput_blob.H = src->H;\n")
    f.write("\tinput_blob.W = src->W;\n")
    f.write("\tinput_blob.dim = src->dim;\n")
    f.write("\tIN_DIFF = BUFF + input_blob.dim;\n")
    f.write("\tW_DATA = BUFF + 2*input_blob.dim;\n")
    f.write("\tupdate_blob();}\n")

    f.write("\nvoid get_output_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\toutput_blob.C = src->C;\n")
    f.write("\toutput_blob.H = src->H;\n")
    f.write("\toutput_blob.W = src->W;\n")
    f.write("\toutput_blob.dim = src->dim;\n")
    f.write("\tOUT_DIFF = BUFF + 2*weight_blob.dim + 2*input_blob.dim + output_blob.dim;\n")
    f.write("\tupdate_blob();}\n")

    f.write("\nvoid get_weight_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\tweight_blob.C = src->C;\n")
    f.write("\tweight_blob.H = src->H;\n")
    f.write("\tweight_blob.W = src->W;\n")
    f.write("\tweight_blob.dim = src->dim;\n")
    f.write("\tW_DIFF = BUFF + weight_blob.dim + 2*input_blob.dim;\n")
    f.write("\tOUT_DATA = BUFF + 2*weight_blob.dim + 2*input_blob.dim;\n")
    f.write("\tupdate_blob();}\n")
   
    f.write("\nvoid copy_struct_param(unsigned int from, unsigned int to, int size){\n")
    f.write("\tpi_cl_dma_cmd(from, to, size, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);}\n")

    f.write("\nvoid reset_arguments(){\n")
    f.write("\tlinear_args.output = &output_blob;\n")
    f.write("\tlinear_args.input = &input_blob;\n")
    f.write("\tlinear_args.coeff = &weight_blob;\n")

    f.write("\tconv2d_args.output = &output_blob;\n")
    f.write("\tconv2d_args.input = &input_blob;\n")
    f.write("\tconv2d_args.coeff = &weight_blob;\n")

    f.write("\tPW_args.output = &output_blob;\n")
    f.write("\tPW_args.input = &input_blob;\n")
    f.write("\tPW_args.coeff = &weight_blob;\n")

    f.write("\tDW_args.output = &output_blob;\n")
    f.write("\tDW_args.input = &input_blob;\n")
    f.write("\tDW_args.coeff = &weight_blob;\n")

    f.write("\tact_args.output = &output_blob;\n")
    f.write("\tact_args.input = &input_blob;\n")

    f.write("\tresconn_args.output = &output_blob;\n")
    f.write("\tresconn_args.lout = &input_blob;\n")
    f.write("\tresconn_args.skip = &weight_blob;\n")
    f.write("}\n\n")

    f.write("\nvoid update_blob(){\n")
    f.write("\tinput_blob.data = IN_DATA;\n")
    f.write("\tinput_blob.diff = IN_DIFF;\n")
    f.write("\toutput_blob.data = OUT_DATA;\n")
    f.write("\toutput_blob.diff = OUT_DIFF;\n")
    f.write("\tweight_blob.data = W_DATA;\n")
    f.write("\tweight_blob.diff = W_DIFF;}\n")

    f.write("\nvoid reset_dim(){\n")
    f.write("\tinput_blob.dim = 0;\n")
    f.write("\tweight_blob.dim = 0;\n")
    f.write("\toutput_blob.dim = 0;}\n")

    f.close()



    return

