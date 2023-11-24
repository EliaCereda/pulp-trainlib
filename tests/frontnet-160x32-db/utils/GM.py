import torch
import torch.nn as nn
import torch.optim as optim
import dump_utils as dump
import math

# Define hyperparameters
learning_rate = 0.01
batch_size = 1
epochs = 5

# LAYER 0 SIZES
l0_in_ch = 1
l0_out_ch = 32
l0_hk = 5
l0_wk = 5
l0_hin = 96
l0_win = 160
l0_hstr = 2
l0_wstr = 2
l0_hpad = 2
l0_wpad = 2
# LAYER 1 SIZES
l1_in_ch = 32
l1_out_ch = 32
l1_hk = 1
l1_wk = 1
l1_hin = 48
l1_win = 80
l1_hstr = 1
l1_wstr = 1
l1_hpad = 0
l1_wpad = 0
# LAYER 2 SIZES
l2_in_ch = 32
l2_out_ch = 32
l2_hk = 1
l2_wk = 1
l2_hin = 48
l2_win = 80
l2_hstr = 1
l2_wstr = 1
l2_hpad = 0
l2_wpad = 0
# LAYER 3 SIZES
l3_in_ch = 32
l3_out_ch = 32
l3_hk = 2
l3_wk = 2
l3_hin = 48
l3_win = 80
l3_hstr = 2
l3_wstr = 2
l3_hpad = 0
l3_wpad = 0
# LAYER 4 SIZES
l4_in_ch = 32
l4_out_ch = 32
l4_hk = 3
l4_wk = 3
l4_hin = 24
l4_win = 40
l4_hstr = 2
l4_wstr = 2
l4_hpad = 1
l4_wpad = 1
# LAYER 5 SIZES
l5_in_ch = 32
l5_out_ch = 32
l5_hk = 1
l5_wk = 1
l5_hin = 12
l5_win = 20
l5_hstr = 1
l5_wstr = 1
l5_hpad = 0
l5_wpad = 0
# LAYER 6 SIZES
l6_in_ch = 32
l6_out_ch = 32
l6_hk = 1
l6_wk = 1
l6_hin = 12
l6_win = 20
l6_hstr = 1
l6_wstr = 1
l6_hpad = 0
l6_wpad = 0
# LAYER 7 SIZES
l7_in_ch = 32
l7_out_ch = 32
l7_hk = 3
l7_wk = 3
l7_hin = 12
l7_win = 20
l7_hstr = 1
l7_wstr = 1
l7_hpad = 1
l7_wpad = 1
# LAYER 8 SIZES
l8_in_ch = 32
l8_out_ch = 32
l8_hk = 1
l8_wk = 1
l8_hin = 12
l8_win = 20
l8_hstr = 1
l8_wstr = 1
l8_hpad = 0
l8_wpad = 0
# LAYER 9 SIZES
l9_in_ch = 32
l9_out_ch = 32
l9_hk = 1
l9_wk = 1
l9_hin = 12
l9_win = 20
l9_hstr = 1
l9_wstr = 1
l9_hpad = 0
l9_wpad = 0
# LAYER 10 SIZES
l10_in_ch = 32
l10_out_ch = 64
l10_hk = 3
l10_wk = 3
l10_hin = 12
l10_win = 20
l10_hstr = 2
l10_wstr = 2
l10_hpad = 1
l10_wpad = 1
# LAYER 11 SIZES
l11_in_ch = 64
l11_out_ch = 64
l11_hk = 1
l11_wk = 1
l11_hin = 6
l11_win = 10
l11_hstr = 1
l11_wstr = 1
l11_hpad = 0
l11_wpad = 0
# LAYER 12 SIZES
l12_in_ch = 64
l12_out_ch = 64
l12_hk = 1
l12_wk = 1
l12_hin = 6
l12_win = 10
l12_hstr = 1
l12_wstr = 1
l12_hpad = 0
l12_wpad = 0
# LAYER 13 SIZES
l13_in_ch = 64
l13_out_ch = 64
l13_hk = 3
l13_wk = 3
l13_hin = 6
l13_win = 10
l13_hstr = 1
l13_wstr = 1
l13_hpad = 1
l13_wpad = 1
# LAYER 14 SIZES
l14_in_ch = 64
l14_out_ch = 64
l14_hk = 1
l14_wk = 1
l14_hin = 6
l14_win = 10
l14_hstr = 1
l14_wstr = 1
l14_hpad = 0
l14_wpad = 0
# LAYER 15 SIZES
l15_in_ch = 64
l15_out_ch = 64
l15_hk = 1
l15_wk = 1
l15_hin = 6
l15_win = 10
l15_hstr = 1
l15_wstr = 1
l15_hpad = 0
l15_wpad = 0
# LAYER 16 SIZES
l16_in_ch = 64
l16_out_ch = 128
l16_hk = 3
l16_wk = 3
l16_hin = 6
l16_win = 10
l16_hstr = 2
l16_wstr = 2
l16_hpad = 1
l16_wpad = 1
# LAYER 17 SIZES
l17_in_ch = 128
l17_out_ch = 128
l17_hk = 1
l17_wk = 1
l17_hin = 3
l17_win = 5
l17_hstr = 1
l17_wstr = 1
l17_hpad = 0
l17_wpad = 0
# LAYER 18 SIZES
l18_in_ch = 128
l18_out_ch = 128
l18_hk = 1
l18_wk = 1
l18_hin = 3
l18_win = 5
l18_hstr = 1
l18_wstr = 1
l18_hpad = 0
l18_wpad = 0
# LAYER 19 SIZES
l19_in_ch = 128
l19_out_ch = 128
l19_hk = 3
l19_wk = 3
l19_hin = 3
l19_win = 5
l19_hstr = 1
l19_wstr = 1
l19_hpad = 1
l19_wpad = 1
# LAYER 20 SIZES
l20_in_ch = 128
l20_out_ch = 128
l20_hk = 1
l20_wk = 1
l20_hin = 3
l20_win = 5
l20_hstr = 1
l20_wstr = 1
l20_hpad = 0
l20_wpad = 0
# LAYER 21 SIZES
l21_in_ch = 128
l21_out_ch = 128
l21_hk = 1
l21_wk = 1
l21_hin = 3
l21_win = 5
l21_hstr = 1
l21_wstr = 1
l21_hpad = 0
l21_wpad = 0
# LAYER 22 SIZES
l22_in_ch = 1920
l22_out_ch = 4
l22_hk = 1
l22_wk = 1
l22_hin = 1
l22_win = 1
l22_hstr = 1
l22_wstr = 1
l22_hpad = 0
l22_wpad = 0

f = open('init-defines.h', 'w')
f.write('// Layer0\n')
f.write('#define Tin_C_l0 '+str(l0_in_ch)+'\n')
f.write('#define Tout_C_l0 '+str(l0_out_ch)+'\n')
f.write('#define Tker_H_l0 '+str(l0_hk)+'\n')
f.write('#define Tker_W_l0 '+str(l0_wk)+'\n')
f.write('#define Tin_H_l0 '+str(l0_hin)+'\n')
f.write('#define Tin_W_l0 '+str(l0_win)+'\n')
f.write('#define Tout_H_l0 '+str(math.floor((l0_hin-l0_hk+2*l0_hpad+l0_hstr)/l0_hstr))+'\n')
f.write('#define Tout_W_l0 '+str(math.floor((l0_win-l0_wk+2*l0_wpad+l0_wstr)/l0_wstr))+'\n')
f.write('#define Tstr_H_l0 '+str(l0_hstr)+'\n')
f.write('#define Tstr_W_l0 '+str(l0_wstr)+'\n')
f.write('#define Tpad_H_l0 '+str(l0_hpad)+'\n')
f.write('#define Tpad_W_l0 '+str(l0_wpad)+'\n')
f.write('// Layer1\n')
f.write('#define Tin_C_l1 '+str(l1_in_ch)+'\n')
f.write('#define Tout_C_l1 '+str(l1_out_ch)+'\n')
f.write('#define Tker_H_l1 '+str(l1_hk)+'\n')
f.write('#define Tker_W_l1 '+str(l1_wk)+'\n')
f.write('#define Tin_H_l1 '+str(l1_hin)+'\n')
f.write('#define Tin_W_l1 '+str(l1_win)+'\n')
f.write('#define Tout_H_l1 '+str(math.floor((l1_hin-l1_hk+2*l1_hpad+l1_hstr)/l1_hstr))+'\n')
f.write('#define Tout_W_l1 '+str(math.floor((l1_win-l1_wk+2*l1_wpad+l1_wstr)/l1_wstr))+'\n')
f.write('#define Tstr_H_l1 '+str(l1_hstr)+'\n')
f.write('#define Tstr_W_l1 '+str(l1_wstr)+'\n')
f.write('#define Tpad_H_l1 '+str(l1_hpad)+'\n')
f.write('#define Tpad_W_l1 '+str(l1_wpad)+'\n')
f.write('// Layer2\n')
f.write('#define Tin_C_l2 '+str(l2_in_ch)+'\n')
f.write('#define Tout_C_l2 '+str(l2_out_ch)+'\n')
f.write('#define Tker_H_l2 '+str(l2_hk)+'\n')
f.write('#define Tker_W_l2 '+str(l2_wk)+'\n')
f.write('#define Tin_H_l2 '+str(l2_hin)+'\n')
f.write('#define Tin_W_l2 '+str(l2_win)+'\n')
f.write('#define Tout_H_l2 '+str(math.floor((l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr))+'\n')
f.write('#define Tout_W_l2 '+str(math.floor((l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr))+'\n')
f.write('#define Tstr_H_l2 '+str(l2_hstr)+'\n')
f.write('#define Tstr_W_l2 '+str(l2_wstr)+'\n')
f.write('#define Tpad_H_l2 '+str(l2_hpad)+'\n')
f.write('#define Tpad_W_l2 '+str(l2_wpad)+'\n')
f.write('// Layer3\n')
f.write('#define Tin_C_l3 '+str(l3_in_ch)+'\n')
f.write('#define Tout_C_l3 '+str(l3_out_ch)+'\n')
f.write('#define Tker_H_l3 '+str(l3_hk)+'\n')
f.write('#define Tker_W_l3 '+str(l3_wk)+'\n')
f.write('#define Tin_H_l3 '+str(l3_hin)+'\n')
f.write('#define Tin_W_l3 '+str(l3_win)+'\n')
f.write('#define Tout_H_l3 '+str(math.floor((l3_hin-l3_hk+2*l3_hpad+l3_hstr)/l3_hstr))+'\n')
f.write('#define Tout_W_l3 '+str(math.floor((l3_win-l3_wk+2*l3_wpad+l3_wstr)/l3_wstr))+'\n')
f.write('#define Tstr_H_l3 '+str(l3_hstr)+'\n')
f.write('#define Tstr_W_l3 '+str(l3_wstr)+'\n')
f.write('#define Tpad_H_l3 '+str(l3_hpad)+'\n')
f.write('#define Tpad_W_l3 '+str(l3_wpad)+'\n')
f.write('// Layer4\n')
f.write('#define Tin_C_l4 '+str(l4_in_ch)+'\n')
f.write('#define Tout_C_l4 '+str(l4_out_ch)+'\n')
f.write('#define Tker_H_l4 '+str(l4_hk)+'\n')
f.write('#define Tker_W_l4 '+str(l4_wk)+'\n')
f.write('#define Tin_H_l4 '+str(l4_hin)+'\n')
f.write('#define Tin_W_l4 '+str(l4_win)+'\n')
f.write('#define Tout_H_l4 '+str(math.floor((l4_hin-l4_hk+2*l4_hpad+l4_hstr)/l4_hstr))+'\n')
f.write('#define Tout_W_l4 '+str(math.floor((l4_win-l4_wk+2*l4_wpad+l4_wstr)/l4_wstr))+'\n')
f.write('#define Tstr_H_l4 '+str(l4_hstr)+'\n')
f.write('#define Tstr_W_l4 '+str(l4_wstr)+'\n')
f.write('#define Tpad_H_l4 '+str(l4_hpad)+'\n')
f.write('#define Tpad_W_l4 '+str(l4_wpad)+'\n')
f.write('// Layer5\n')
f.write('#define Tin_C_l5 '+str(l5_in_ch)+'\n')
f.write('#define Tout_C_l5 '+str(l5_out_ch)+'\n')
f.write('#define Tker_H_l5 '+str(l5_hk)+'\n')
f.write('#define Tker_W_l5 '+str(l5_wk)+'\n')
f.write('#define Tin_H_l5 '+str(l5_hin)+'\n')
f.write('#define Tin_W_l5 '+str(l5_win)+'\n')
f.write('#define Tout_H_l5 '+str(math.floor((l5_hin-l5_hk+2*l5_hpad+l5_hstr)/l5_hstr))+'\n')
f.write('#define Tout_W_l5 '+str(math.floor((l5_win-l5_wk+2*l5_wpad+l5_wstr)/l5_wstr))+'\n')
f.write('#define Tstr_H_l5 '+str(l5_hstr)+'\n')
f.write('#define Tstr_W_l5 '+str(l5_wstr)+'\n')
f.write('#define Tpad_H_l5 '+str(l5_hpad)+'\n')
f.write('#define Tpad_W_l5 '+str(l5_wpad)+'\n')
f.write('// Layer6\n')
f.write('#define Tin_C_l6 '+str(l6_in_ch)+'\n')
f.write('#define Tout_C_l6 '+str(l6_out_ch)+'\n')
f.write('#define Tker_H_l6 '+str(l6_hk)+'\n')
f.write('#define Tker_W_l6 '+str(l6_wk)+'\n')
f.write('#define Tin_H_l6 '+str(l6_hin)+'\n')
f.write('#define Tin_W_l6 '+str(l6_win)+'\n')
f.write('#define Tout_H_l6 '+str(math.floor((l6_hin-l6_hk+2*l6_hpad+l6_hstr)/l6_hstr))+'\n')
f.write('#define Tout_W_l6 '+str(math.floor((l6_win-l6_wk+2*l6_wpad+l6_wstr)/l6_wstr))+'\n')
f.write('#define Tstr_H_l6 '+str(l6_hstr)+'\n')
f.write('#define Tstr_W_l6 '+str(l6_wstr)+'\n')
f.write('#define Tpad_H_l6 '+str(l6_hpad)+'\n')
f.write('#define Tpad_W_l6 '+str(l6_wpad)+'\n')
f.write('// Layer7\n')
f.write('#define Tin_C_l7 '+str(l7_in_ch)+'\n')
f.write('#define Tout_C_l7 '+str(l7_out_ch)+'\n')
f.write('#define Tker_H_l7 '+str(l7_hk)+'\n')
f.write('#define Tker_W_l7 '+str(l7_wk)+'\n')
f.write('#define Tin_H_l7 '+str(l7_hin)+'\n')
f.write('#define Tin_W_l7 '+str(l7_win)+'\n')
f.write('#define Tout_H_l7 '+str(math.floor((l7_hin-l7_hk+2*l7_hpad+l7_hstr)/l7_hstr))+'\n')
f.write('#define Tout_W_l7 '+str(math.floor((l7_win-l7_wk+2*l7_wpad+l7_wstr)/l7_wstr))+'\n')
f.write('#define Tstr_H_l7 '+str(l7_hstr)+'\n')
f.write('#define Tstr_W_l7 '+str(l7_wstr)+'\n')
f.write('#define Tpad_H_l7 '+str(l7_hpad)+'\n')
f.write('#define Tpad_W_l7 '+str(l7_wpad)+'\n')
f.write('// Layer8\n')
f.write('#define Tin_C_l8 '+str(l8_in_ch)+'\n')
f.write('#define Tout_C_l8 '+str(l8_out_ch)+'\n')
f.write('#define Tker_H_l8 '+str(l8_hk)+'\n')
f.write('#define Tker_W_l8 '+str(l8_wk)+'\n')
f.write('#define Tin_H_l8 '+str(l8_hin)+'\n')
f.write('#define Tin_W_l8 '+str(l8_win)+'\n')
f.write('#define Tout_H_l8 '+str(math.floor((l8_hin-l8_hk+2*l8_hpad+l8_hstr)/l8_hstr))+'\n')
f.write('#define Tout_W_l8 '+str(math.floor((l8_win-l8_wk+2*l8_wpad+l8_wstr)/l8_wstr))+'\n')
f.write('#define Tstr_H_l8 '+str(l8_hstr)+'\n')
f.write('#define Tstr_W_l8 '+str(l8_wstr)+'\n')
f.write('#define Tpad_H_l8 '+str(l8_hpad)+'\n')
f.write('#define Tpad_W_l8 '+str(l8_wpad)+'\n')
f.write('// Layer9\n')
f.write('#define Tin_C_l9 '+str(l9_in_ch)+'\n')
f.write('#define Tout_C_l9 '+str(l9_out_ch)+'\n')
f.write('#define Tker_H_l9 '+str(l9_hk)+'\n')
f.write('#define Tker_W_l9 '+str(l9_wk)+'\n')
f.write('#define Tin_H_l9 '+str(l9_hin)+'\n')
f.write('#define Tin_W_l9 '+str(l9_win)+'\n')
f.write('#define Tout_H_l9 '+str(math.floor((l9_hin-l9_hk+2*l9_hpad+l9_hstr)/l9_hstr))+'\n')
f.write('#define Tout_W_l9 '+str(math.floor((l9_win-l9_wk+2*l9_wpad+l9_wstr)/l9_wstr))+'\n')
f.write('#define Tstr_H_l9 '+str(l9_hstr)+'\n')
f.write('#define Tstr_W_l9 '+str(l9_wstr)+'\n')
f.write('#define Tpad_H_l9 '+str(l9_hpad)+'\n')
f.write('#define Tpad_W_l9 '+str(l9_wpad)+'\n')
f.write('// Layer10\n')
f.write('#define Tin_C_l10 '+str(l10_in_ch)+'\n')
f.write('#define Tout_C_l10 '+str(l10_out_ch)+'\n')
f.write('#define Tker_H_l10 '+str(l10_hk)+'\n')
f.write('#define Tker_W_l10 '+str(l10_wk)+'\n')
f.write('#define Tin_H_l10 '+str(l10_hin)+'\n')
f.write('#define Tin_W_l10 '+str(l10_win)+'\n')
f.write('#define Tout_H_l10 '+str(math.floor((l10_hin-l10_hk+2*l10_hpad+l10_hstr)/l10_hstr))+'\n')
f.write('#define Tout_W_l10 '+str(math.floor((l10_win-l10_wk+2*l10_wpad+l10_wstr)/l10_wstr))+'\n')
f.write('#define Tstr_H_l10 '+str(l10_hstr)+'\n')
f.write('#define Tstr_W_l10 '+str(l10_wstr)+'\n')
f.write('#define Tpad_H_l10 '+str(l10_hpad)+'\n')
f.write('#define Tpad_W_l10 '+str(l10_wpad)+'\n')
f.write('// Layer11\n')
f.write('#define Tin_C_l11 '+str(l11_in_ch)+'\n')
f.write('#define Tout_C_l11 '+str(l11_out_ch)+'\n')
f.write('#define Tker_H_l11 '+str(l11_hk)+'\n')
f.write('#define Tker_W_l11 '+str(l11_wk)+'\n')
f.write('#define Tin_H_l11 '+str(l11_hin)+'\n')
f.write('#define Tin_W_l11 '+str(l11_win)+'\n')
f.write('#define Tout_H_l11 '+str(math.floor((l11_hin-l11_hk+2*l11_hpad+l11_hstr)/l11_hstr))+'\n')
f.write('#define Tout_W_l11 '+str(math.floor((l11_win-l11_wk+2*l11_wpad+l11_wstr)/l11_wstr))+'\n')
f.write('#define Tstr_H_l11 '+str(l11_hstr)+'\n')
f.write('#define Tstr_W_l11 '+str(l11_wstr)+'\n')
f.write('#define Tpad_H_l11 '+str(l11_hpad)+'\n')
f.write('#define Tpad_W_l11 '+str(l11_wpad)+'\n')
f.write('// Layer12\n')
f.write('#define Tin_C_l12 '+str(l12_in_ch)+'\n')
f.write('#define Tout_C_l12 '+str(l12_out_ch)+'\n')
f.write('#define Tker_H_l12 '+str(l12_hk)+'\n')
f.write('#define Tker_W_l12 '+str(l12_wk)+'\n')
f.write('#define Tin_H_l12 '+str(l12_hin)+'\n')
f.write('#define Tin_W_l12 '+str(l12_win)+'\n')
f.write('#define Tout_H_l12 '+str(math.floor((l12_hin-l12_hk+2*l12_hpad+l12_hstr)/l12_hstr))+'\n')
f.write('#define Tout_W_l12 '+str(math.floor((l12_win-l12_wk+2*l12_wpad+l12_wstr)/l12_wstr))+'\n')
f.write('#define Tstr_H_l12 '+str(l12_hstr)+'\n')
f.write('#define Tstr_W_l12 '+str(l12_wstr)+'\n')
f.write('#define Tpad_H_l12 '+str(l12_hpad)+'\n')
f.write('#define Tpad_W_l12 '+str(l12_wpad)+'\n')
f.write('// Layer13\n')
f.write('#define Tin_C_l13 '+str(l13_in_ch)+'\n')
f.write('#define Tout_C_l13 '+str(l13_out_ch)+'\n')
f.write('#define Tker_H_l13 '+str(l13_hk)+'\n')
f.write('#define Tker_W_l13 '+str(l13_wk)+'\n')
f.write('#define Tin_H_l13 '+str(l13_hin)+'\n')
f.write('#define Tin_W_l13 '+str(l13_win)+'\n')
f.write('#define Tout_H_l13 '+str(math.floor((l13_hin-l13_hk+2*l13_hpad+l13_hstr)/l13_hstr))+'\n')
f.write('#define Tout_W_l13 '+str(math.floor((l13_win-l13_wk+2*l13_wpad+l13_wstr)/l13_wstr))+'\n')
f.write('#define Tstr_H_l13 '+str(l13_hstr)+'\n')
f.write('#define Tstr_W_l13 '+str(l13_wstr)+'\n')
f.write('#define Tpad_H_l13 '+str(l13_hpad)+'\n')
f.write('#define Tpad_W_l13 '+str(l13_wpad)+'\n')
f.write('// Layer14\n')
f.write('#define Tin_C_l14 '+str(l14_in_ch)+'\n')
f.write('#define Tout_C_l14 '+str(l14_out_ch)+'\n')
f.write('#define Tker_H_l14 '+str(l14_hk)+'\n')
f.write('#define Tker_W_l14 '+str(l14_wk)+'\n')
f.write('#define Tin_H_l14 '+str(l14_hin)+'\n')
f.write('#define Tin_W_l14 '+str(l14_win)+'\n')
f.write('#define Tout_H_l14 '+str(math.floor((l14_hin-l14_hk+2*l14_hpad+l14_hstr)/l14_hstr))+'\n')
f.write('#define Tout_W_l14 '+str(math.floor((l14_win-l14_wk+2*l14_wpad+l14_wstr)/l14_wstr))+'\n')
f.write('#define Tstr_H_l14 '+str(l14_hstr)+'\n')
f.write('#define Tstr_W_l14 '+str(l14_wstr)+'\n')
f.write('#define Tpad_H_l14 '+str(l14_hpad)+'\n')
f.write('#define Tpad_W_l14 '+str(l14_wpad)+'\n')
f.write('// Layer15\n')
f.write('#define Tin_C_l15 '+str(l15_in_ch)+'\n')
f.write('#define Tout_C_l15 '+str(l15_out_ch)+'\n')
f.write('#define Tker_H_l15 '+str(l15_hk)+'\n')
f.write('#define Tker_W_l15 '+str(l15_wk)+'\n')
f.write('#define Tin_H_l15 '+str(l15_hin)+'\n')
f.write('#define Tin_W_l15 '+str(l15_win)+'\n')
f.write('#define Tout_H_l15 '+str(math.floor((l15_hin-l15_hk+2*l15_hpad+l15_hstr)/l15_hstr))+'\n')
f.write('#define Tout_W_l15 '+str(math.floor((l15_win-l15_wk+2*l15_wpad+l15_wstr)/l15_wstr))+'\n')
f.write('#define Tstr_H_l15 '+str(l15_hstr)+'\n')
f.write('#define Tstr_W_l15 '+str(l15_wstr)+'\n')
f.write('#define Tpad_H_l15 '+str(l15_hpad)+'\n')
f.write('#define Tpad_W_l15 '+str(l15_wpad)+'\n')
f.write('// Layer16\n')
f.write('#define Tin_C_l16 '+str(l16_in_ch)+'\n')
f.write('#define Tout_C_l16 '+str(l16_out_ch)+'\n')
f.write('#define Tker_H_l16 '+str(l16_hk)+'\n')
f.write('#define Tker_W_l16 '+str(l16_wk)+'\n')
f.write('#define Tin_H_l16 '+str(l16_hin)+'\n')
f.write('#define Tin_W_l16 '+str(l16_win)+'\n')
f.write('#define Tout_H_l16 '+str(math.floor((l16_hin-l16_hk+2*l16_hpad+l16_hstr)/l16_hstr))+'\n')
f.write('#define Tout_W_l16 '+str(math.floor((l16_win-l16_wk+2*l16_wpad+l16_wstr)/l16_wstr))+'\n')
f.write('#define Tstr_H_l16 '+str(l16_hstr)+'\n')
f.write('#define Tstr_W_l16 '+str(l16_wstr)+'\n')
f.write('#define Tpad_H_l16 '+str(l16_hpad)+'\n')
f.write('#define Tpad_W_l16 '+str(l16_wpad)+'\n')
f.write('// Layer17\n')
f.write('#define Tin_C_l17 '+str(l17_in_ch)+'\n')
f.write('#define Tout_C_l17 '+str(l17_out_ch)+'\n')
f.write('#define Tker_H_l17 '+str(l17_hk)+'\n')
f.write('#define Tker_W_l17 '+str(l17_wk)+'\n')
f.write('#define Tin_H_l17 '+str(l17_hin)+'\n')
f.write('#define Tin_W_l17 '+str(l17_win)+'\n')
f.write('#define Tout_H_l17 '+str(math.floor((l17_hin-l17_hk+2*l17_hpad+l17_hstr)/l17_hstr))+'\n')
f.write('#define Tout_W_l17 '+str(math.floor((l17_win-l17_wk+2*l17_wpad+l17_wstr)/l17_wstr))+'\n')
f.write('#define Tstr_H_l17 '+str(l17_hstr)+'\n')
f.write('#define Tstr_W_l17 '+str(l17_wstr)+'\n')
f.write('#define Tpad_H_l17 '+str(l17_hpad)+'\n')
f.write('#define Tpad_W_l17 '+str(l17_wpad)+'\n')
f.write('// Layer18\n')
f.write('#define Tin_C_l18 '+str(l18_in_ch)+'\n')
f.write('#define Tout_C_l18 '+str(l18_out_ch)+'\n')
f.write('#define Tker_H_l18 '+str(l18_hk)+'\n')
f.write('#define Tker_W_l18 '+str(l18_wk)+'\n')
f.write('#define Tin_H_l18 '+str(l18_hin)+'\n')
f.write('#define Tin_W_l18 '+str(l18_win)+'\n')
f.write('#define Tout_H_l18 '+str(math.floor((l18_hin-l18_hk+2*l18_hpad+l18_hstr)/l18_hstr))+'\n')
f.write('#define Tout_W_l18 '+str(math.floor((l18_win-l18_wk+2*l18_wpad+l18_wstr)/l18_wstr))+'\n')
f.write('#define Tstr_H_l18 '+str(l18_hstr)+'\n')
f.write('#define Tstr_W_l18 '+str(l18_wstr)+'\n')
f.write('#define Tpad_H_l18 '+str(l18_hpad)+'\n')
f.write('#define Tpad_W_l18 '+str(l18_wpad)+'\n')
f.write('// Layer19\n')
f.write('#define Tin_C_l19 '+str(l19_in_ch)+'\n')
f.write('#define Tout_C_l19 '+str(l19_out_ch)+'\n')
f.write('#define Tker_H_l19 '+str(l19_hk)+'\n')
f.write('#define Tker_W_l19 '+str(l19_wk)+'\n')
f.write('#define Tin_H_l19 '+str(l19_hin)+'\n')
f.write('#define Tin_W_l19 '+str(l19_win)+'\n')
f.write('#define Tout_H_l19 '+str(math.floor((l19_hin-l19_hk+2*l19_hpad+l19_hstr)/l19_hstr))+'\n')
f.write('#define Tout_W_l19 '+str(math.floor((l19_win-l19_wk+2*l19_wpad+l19_wstr)/l19_wstr))+'\n')
f.write('#define Tstr_H_l19 '+str(l19_hstr)+'\n')
f.write('#define Tstr_W_l19 '+str(l19_wstr)+'\n')
f.write('#define Tpad_H_l19 '+str(l19_hpad)+'\n')
f.write('#define Tpad_W_l19 '+str(l19_wpad)+'\n')
f.write('// Layer20\n')
f.write('#define Tin_C_l20 '+str(l20_in_ch)+'\n')
f.write('#define Tout_C_l20 '+str(l20_out_ch)+'\n')
f.write('#define Tker_H_l20 '+str(l20_hk)+'\n')
f.write('#define Tker_W_l20 '+str(l20_wk)+'\n')
f.write('#define Tin_H_l20 '+str(l20_hin)+'\n')
f.write('#define Tin_W_l20 '+str(l20_win)+'\n')
f.write('#define Tout_H_l20 '+str(math.floor((l20_hin-l20_hk+2*l20_hpad+l20_hstr)/l20_hstr))+'\n')
f.write('#define Tout_W_l20 '+str(math.floor((l20_win-l20_wk+2*l20_wpad+l20_wstr)/l20_wstr))+'\n')
f.write('#define Tstr_H_l20 '+str(l20_hstr)+'\n')
f.write('#define Tstr_W_l20 '+str(l20_wstr)+'\n')
f.write('#define Tpad_H_l20 '+str(l20_hpad)+'\n')
f.write('#define Tpad_W_l20 '+str(l20_wpad)+'\n')
f.write('// Layer21\n')
f.write('#define Tin_C_l21 '+str(l21_in_ch)+'\n')
f.write('#define Tout_C_l21 '+str(l21_out_ch)+'\n')
f.write('#define Tker_H_l21 '+str(l21_hk)+'\n')
f.write('#define Tker_W_l21 '+str(l21_wk)+'\n')
f.write('#define Tin_H_l21 '+str(l21_hin)+'\n')
f.write('#define Tin_W_l21 '+str(l21_win)+'\n')
f.write('#define Tout_H_l21 '+str(math.floor((l21_hin-l21_hk+2*l21_hpad+l21_hstr)/l21_hstr))+'\n')
f.write('#define Tout_W_l21 '+str(math.floor((l21_win-l21_wk+2*l21_wpad+l21_wstr)/l21_wstr))+'\n')
f.write('#define Tstr_H_l21 '+str(l21_hstr)+'\n')
f.write('#define Tstr_W_l21 '+str(l21_wstr)+'\n')
f.write('#define Tpad_H_l21 '+str(l21_hpad)+'\n')
f.write('#define Tpad_W_l21 '+str(l21_wpad)+'\n')
f.write('// Layer22\n')
f.write('#define Tin_C_l22 '+str(l22_in_ch)+'\n')
f.write('#define Tout_C_l22 '+str(l22_out_ch)+'\n')
f.write('#define Tker_H_l22 '+str(l22_hk)+'\n')
f.write('#define Tker_W_l22 '+str(l22_wk)+'\n')
f.write('#define Tin_H_l22 '+str(l22_hin)+'\n')
f.write('#define Tin_W_l22 '+str(l22_win)+'\n')
f.write('#define Tout_H_l22 '+str(math.floor((l22_hin-l22_hk+2*l22_hpad+l22_hstr)/l22_hstr))+'\n')
f.write('#define Tout_W_l22 '+str(math.floor((l22_win-l22_wk+2*l22_wpad+l22_wstr)/l22_wstr))+'\n')
f.write('#define Tstr_H_l22 '+str(l22_hstr)+'\n')
f.write('#define Tstr_W_l22 '+str(l22_wstr)+'\n')
f.write('#define Tpad_H_l22 '+str(l22_hpad)+'\n')
f.write('#define Tpad_W_l22 '+str(l22_wpad)+'\n')
f.close()

f = open('init-defines.h', 'a')
f.write('\n// HYPERPARAMETERS\n')
f.write('#define LEARNING_RATE '+str(learning_rate)+'\n')
f.write('#define EPOCHS '+str(epochs)+'\n')
f.write('#define BATCH_SIZE '+str(batch_size)+'\n')
f.close()


# Simple input data 
inp = torch.torch.div(torch.randint(1000, [batch_size, l0_in_ch, l0_hin, l0_win]), 1000)

class Sumnode():
	def __init__(self, ls):
		self.MySkipNode = ls

class Skipnode():
	def __init__(self):
		self.data = 0

	def __call__(self, x):
		self.data = x
		return self.data

class DNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.l0 = nn.Conv2d(in_channels=l0_in_ch, out_channels=l0_out_ch, kernel_size=(l0_hk, l0_wk), padding=(l0_hpad, l0_wpad), stride=(l0_hstr, l0_wstr), bias=False)
		self.l1= nn.InstanceNorm2d(num_features=32, eps=1e-10, momentum=0, affine=True)
		self.l2 = nn.ReLU()
		self.l3 = nn.MaxPool2d(kernel_size=(l3_hk, l3_wk), stride=(l3_hstr, l3_wstr))
		self.l4 = nn.Conv2d(in_channels=l4_in_ch, out_channels=l4_out_ch, kernel_size=(l4_hk, l4_wk), padding=(l4_hpad, l4_wpad), stride=(l4_hstr, l4_wstr), bias=False)
		self.l5= nn.InstanceNorm2d(num_features=32, eps=1e-10, momentum=0, affine=True)
		self.l6 = nn.ReLU()
		self.l7 = nn.Conv2d(in_channels=l7_in_ch, out_channels=l7_out_ch, kernel_size=(l7_hk, l7_wk), padding=(l7_hpad, l7_wpad), stride=(l7_hstr, l7_wstr), bias=False)
		self.l8= nn.InstanceNorm2d(num_features=32, eps=1e-10, momentum=0, affine=True)
		self.l9 = nn.ReLU()
		self.l10 = nn.Conv2d(in_channels=l10_in_ch, out_channels=l10_out_ch, kernel_size=(l10_hk, l10_wk), padding=(l10_hpad, l10_wpad), stride=(l10_hstr, l10_wstr), bias=False)
		self.l11= nn.InstanceNorm2d(num_features=64, eps=1e-10, momentum=0, affine=True)
		self.l12 = nn.ReLU()
		self.l13 = nn.Conv2d(in_channels=l13_in_ch, out_channels=l13_out_ch, kernel_size=(l13_hk, l13_wk), padding=(l13_hpad, l13_wpad), stride=(l13_hstr, l13_wstr), bias=False)
		self.l14= nn.InstanceNorm2d(num_features=64, eps=1e-10, momentum=0, affine=True)
		self.l15 = nn.ReLU()
		self.l16 = nn.Conv2d(in_channels=l16_in_ch, out_channels=l16_out_ch, kernel_size=(l16_hk, l16_wk), padding=(l16_hpad, l16_wpad), stride=(l16_hstr, l16_wstr), bias=False)
		self.l17= nn.InstanceNorm2d(num_features=128, eps=1e-10, momentum=0, affine=True)
		self.l18 = nn.ReLU()
		self.l19 = nn.Conv2d(in_channels=l19_in_ch, out_channels=l19_out_ch, kernel_size=(l19_hk, l19_wk), padding=(l19_hpad, l19_wpad), stride=(l19_hstr, l19_wstr), bias=False)
		self.l20= nn.InstanceNorm2d(num_features=128, eps=1e-10, momentum=0, affine=True)
		self.l21 = nn.ReLU()
		self.l22 = nn.Linear(in_features=l22_in_ch, out_features=l22_out_ch, bias=False)

	def forward(self, x):
		x = self.l0(x)
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = self.l6(x)
		x = self.l7(x)
		x = self.l8(x)
		x = self.l9(x)
		x = self.l10(x)
		x = self.l11(x)
		x = self.l12(x)
		x = self.l13(x)
		x = self.l14(x)
		x = self.l15(x)
		x = self.l16(x)
		x = self.l17(x)
		x = self.l18(x)
		x = self.l19(x)
		x = self.l20(x)
		x = self.l21(x)
		x = torch.reshape(x, (-1,))
		x = self.l22(x).float()
		return x

# Initialize network
net = DNN()
for p in net.parameters():
	nn.init.normal_(p, mean=0.0, std=1.0)
net.zero_grad()


# All-ones fake label 
output_test = net(inp)
label = torch.ones_like(output_test)
f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write('#define WGT_SIZE_L0 '+str(l0_in_ch*l0_out_ch*l0_hk*l0_wk)+'\n')
f.write('PI_L2 float init_WGT_l0[WGT_SIZE_L0] = {'+dump.tensor_to_string(net.l0.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L1  2*{l1_in_ch}\n')
f.write('PI_L2 float init_WGT_l1[WGT_SIZE_L1] = {'+dump.tensor_to_string(net.l1.weight.data)+dump.tensor_to_string(net.l1.bias.data)+'};\n')
f.write('#define WGT_SIZE_L2 '+str(l2_in_ch*l2_out_ch*l2_hk*l2_wk)+'\n')
f.write('PI_L2 float init_WGT_l2[WGT_SIZE_L2];\n')
f.write('#define WGT_SIZE_L3 '+str(l3_in_ch*l3_out_ch*l3_hk*l3_wk)+'\n')
f.write('PI_L2 float init_WGT_l3[WGT_SIZE_L3];\n')
f.write('#define WGT_SIZE_L4 '+str(l4_in_ch*l4_out_ch*l4_hk*l4_wk)+'\n')
f.write('PI_L2 float init_WGT_l4[WGT_SIZE_L4] = {'+dump.tensor_to_string(net.l4.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L5  2*{l5_in_ch}\n')
f.write('PI_L2 float init_WGT_l5[WGT_SIZE_L5] = {'+dump.tensor_to_string(net.l5.weight.data)+dump.tensor_to_string(net.l5.bias.data)+'};\n')
f.write('#define WGT_SIZE_L6 '+str(l6_in_ch*l6_out_ch*l6_hk*l6_wk)+'\n')
f.write('PI_L2 float init_WGT_l6[WGT_SIZE_L6];\n')
f.write('#define WGT_SIZE_L7 '+str(l7_in_ch*l7_out_ch*l7_hk*l7_wk)+'\n')
f.write('PI_L2 float init_WGT_l7[WGT_SIZE_L7] = {'+dump.tensor_to_string(net.l7.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L8  2*{l8_in_ch}\n')
f.write('PI_L2 float init_WGT_l8[WGT_SIZE_L8] = {'+dump.tensor_to_string(net.l8.weight.data)+dump.tensor_to_string(net.l8.bias.data)+'};\n')
f.write('#define WGT_SIZE_L9 '+str(l9_in_ch*l9_out_ch*l9_hk*l9_wk)+'\n')
f.write('PI_L2 float init_WGT_l9[WGT_SIZE_L9];\n')
f.write('#define WGT_SIZE_L10 '+str(l10_in_ch*l10_out_ch*l10_hk*l10_wk)+'\n')
f.write('PI_L2 float init_WGT_l10[WGT_SIZE_L10] = {'+dump.tensor_to_string(net.l10.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L11  2*{l11_in_ch}\n')
f.write('PI_L2 float init_WGT_l11[WGT_SIZE_L11] = {'+dump.tensor_to_string(net.l11.weight.data)+dump.tensor_to_string(net.l11.bias.data)+'};\n')
f.write('#define WGT_SIZE_L12 '+str(l12_in_ch*l12_out_ch*l12_hk*l12_wk)+'\n')
f.write('PI_L2 float init_WGT_l12[WGT_SIZE_L12];\n')
f.write('#define WGT_SIZE_L13 '+str(l13_in_ch*l13_out_ch*l13_hk*l13_wk)+'\n')
f.write('PI_L2 float init_WGT_l13[WGT_SIZE_L13] = {'+dump.tensor_to_string(net.l13.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L14  2*{l14_in_ch}\n')
f.write('PI_L2 float init_WGT_l14[WGT_SIZE_L14] = {'+dump.tensor_to_string(net.l14.weight.data)+dump.tensor_to_string(net.l14.bias.data)+'};\n')
f.write('#define WGT_SIZE_L15 '+str(l15_in_ch*l15_out_ch*l15_hk*l15_wk)+'\n')
f.write('PI_L2 float init_WGT_l15[WGT_SIZE_L15];\n')
f.write('#define WGT_SIZE_L16 '+str(l16_in_ch*l16_out_ch*l16_hk*l16_wk)+'\n')
f.write('PI_L2 float init_WGT_l16[WGT_SIZE_L16] = {'+dump.tensor_to_string(net.l16.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L17  2*{l17_in_ch}\n')
f.write('PI_L2 float init_WGT_l17[WGT_SIZE_L17] = {'+dump.tensor_to_string(net.l17.weight.data)+dump.tensor_to_string(net.l17.bias.data)+'};\n')
f.write('#define WGT_SIZE_L18 '+str(l18_in_ch*l18_out_ch*l18_hk*l18_wk)+'\n')
f.write('PI_L2 float init_WGT_l18[WGT_SIZE_L18];\n')
f.write('#define WGT_SIZE_L19 '+str(l19_in_ch*l19_out_ch*l19_hk*l19_wk)+'\n')
f.write('PI_L2 float init_WGT_l19[WGT_SIZE_L19] = {'+dump.tensor_to_string(net.l19.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L20  2*{l20_in_ch}\n')
f.write('PI_L2 float init_WGT_l20[WGT_SIZE_L20] = {'+dump.tensor_to_string(net.l20.weight.data)+dump.tensor_to_string(net.l20.bias.data)+'};\n')
f.write('#define WGT_SIZE_L21 '+str(l21_in_ch*l21_out_ch*l21_hk*l21_wk)+'\n')
f.write('PI_L2 float init_WGT_l21[WGT_SIZE_L21];\n')
f.write('#define WGT_SIZE_L22 '+str(l22_in_ch*l22_out_ch*l22_hk*l22_wk)+'\n')
f.write('PI_L2 float init_WGT_l22[WGT_SIZE_L22] = {'+dump.tensor_to_string(net.l22.weight.data)+'};\n')
f.close()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
loss_fn = nn.MSELoss()

# Train the DNN
for batch in range(epochs):
	optimizer.zero_grad()
	out = net(inp)
	loss = loss_fn(out, label)
	loss.backward()
	optimizer.step()

# Inference once after training
out = net(inp)

f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write('#define IN_SIZE 15360\n')
f.write('PI_L2 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
out_size = (int(math.floor(l22_hin-l22_hk+2*l22_hpad+l22_hstr)/l22_hstr)) * (int(math.floor(l22_win-l22_wk+2*l22_wpad+l22_wstr)/l22_wstr)) * l22_out_ch
f.write('#define OUT_SIZE '+str(out_size)+'\n')
f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.write('PI_L2 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\n')
f.close()
