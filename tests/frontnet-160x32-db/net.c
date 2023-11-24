/**
 * INCLUDES
**/

#include "pulp_train.h"
#include "net.h"
#include "stats.h"

#include "init-defines.h"
#include "io_data.h"

// Define structures and pointers to data in L1 memory
PI_L1 float BUFF[MAX_SIZE];
PI_L1 struct blob d1_blob;
PI_L1 struct blob w1_blob;
PI_L1 struct blob d0_blob;
PI_L1 struct blob w0_blob;
PI_L1 struct blob in;
PI_L1 struct blob wgt;
PI_L1 struct blob out;
PI_L1 struct Linear_args linear_args;
PI_L1 struct Conv2D_args conv2d_args;
PI_L1 struct PointWise_Conv_args PW_args;
PI_L1 struct DepthWise_Conv_args DW_args;
PI_L1 struct act_args act_args;
PI_L1 struct InstNorm_args InstNorm_args;
PI_L1 struct SkipConn_args resconn_args;
PI_L1 pi_cl_dma_cmd_t * cmd_store;
PI_L1 pi_cl_dma_cmd_t * cmd_load;
PI_L1 pi_cl_dma_cmd_t * cmd_struct;



/**
 * DATA
**/

// Define loss
PI_L1 float loss = 0;

// Define DNN blobs
PI_L2 struct blob layer0_in, layer0_wgt, layer0_out;
PI_L2 struct blob layer1_in, layer1_wgt, layer1_out;
PI_L2 struct blob layer2_in, layer2_wgt, layer2_out;
PI_L2 struct blob layer3_in, layer3_wgt, layer3_out;
PI_L2 struct blob layer4_in, layer4_wgt, layer4_out;
PI_L2 struct blob layer5_in, layer5_wgt, layer5_out;
PI_L2 struct blob layer6_in, layer6_wgt, layer6_out;
PI_L2 struct blob layer7_in, layer7_wgt, layer7_out;
PI_L2 struct blob layer8_in, layer8_wgt, layer8_out;
PI_L2 struct blob layer9_in, layer9_wgt, layer9_out;
PI_L2 struct blob layer10_in, layer10_wgt, layer10_out;
PI_L2 struct blob layer11_in, layer11_wgt, layer11_out;
PI_L2 struct blob layer12_in, layer12_wgt, layer12_out;
PI_L2 struct blob layer13_in, layer13_wgt, layer13_out;
PI_L2 struct blob layer14_in, layer14_wgt, layer14_out;
PI_L2 struct blob layer15_in, layer15_wgt, layer15_out;
PI_L2 struct blob layer16_in, layer16_wgt, layer16_out;
PI_L2 struct blob layer17_in, layer17_wgt, layer17_out;
PI_L2 struct blob layer18_in, layer18_wgt, layer18_out;
PI_L2 struct blob layer19_in, layer19_wgt, layer19_out;
PI_L2 struct blob layer20_in, layer20_wgt, layer20_out;
PI_L2 struct blob layer21_in, layer21_wgt, layer21_out;
PI_L2 struct blob layer22_in, layer22_wgt, layer22_out;

// Define DNN layer structures
PI_L1 struct vect_sum_args vect_sum_args;
PI_L2 struct Conv2D_args l0_args;
PI_L2 struct InstNorm_args l1_args;
PI_L2 struct act_args l2_args;
PI_L2 struct Conv2D_args l4_args;
PI_L2 struct InstNorm_args l5_args;
PI_L2 struct act_args l6_args;
PI_L2 struct Conv2D_args l7_args;
PI_L2 struct InstNorm_args l8_args;
PI_L2 struct act_args l9_args;
PI_L2 struct Conv2D_args l10_args;
PI_L2 struct InstNorm_args l11_args;
PI_L2 struct act_args l12_args;
PI_L2 struct Conv2D_args l13_args;
PI_L2 struct InstNorm_args l14_args;
PI_L2 struct act_args l15_args;
PI_L2 struct Conv2D_args l16_args;
PI_L2 struct InstNorm_args l17_args;
PI_L2 struct act_args l18_args;
PI_L2 struct Conv2D_args l19_args;
PI_L2 struct InstNorm_args l20_args;
PI_L2 struct act_args l21_args;
PI_L2 struct Linear_args l22_args;

// Define Pooling Structures
PI_L2 struct pool_args l3_pool_args;

// Define kernel tensors
PI_L2 float l0_ker[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker[2*Tin_C_l1];
PI_L2 float l2_ker[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];
PI_L2 float l3_ker[1];
PI_L2 float l4_ker[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker[2*Tin_C_l5];
PI_L2 float l6_ker[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L2 float l7_ker[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_ker[2*Tin_C_l8];
PI_L2 float l9_ker[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker[2*Tin_C_l11];
PI_L2 float l12_ker[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker[2*Tin_C_l14];
PI_L2 float l15_ker[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];
PI_L2 float l16_ker[Tin_C_l16 * Tout_C_l16 * Tker_H_l16 * Tker_W_l16];
PI_L2 float l17_ker[2*Tin_C_l17];
PI_L2 float l18_ker[Tin_C_l18 * Tout_C_l18 * Tker_H_l18 * Tker_W_l18];
PI_L2 float l19_ker[Tin_C_l19 * Tout_C_l19 * Tker_H_l19 * Tker_W_l19];
PI_L2 float l20_ker[2*Tin_C_l20];
PI_L2 float l21_ker[Tin_C_l21 * Tout_C_l21 * Tker_H_l21 * Tker_W_l21];
PI_L2 float l22_ker[Tin_C_l22 * Tout_C_l22 * Tker_H_l22 * Tker_W_l22];

// Define kernel grad tensors
PI_L2 float l0_ker_diff[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker_diff[2*Tin_C_l1];
PI_L2 float l2_ker_diff[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];
PI_L2 float l3_ker_diff[1];
PI_L2 float l4_ker_diff[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker_diff[2*Tin_C_l5];
PI_L2 float l6_ker_diff[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L2 float l7_ker_diff[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_ker_diff[2*Tin_C_l8];
PI_L2 float l9_ker_diff[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker_diff[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker_diff[2*Tin_C_l11];
PI_L2 float l12_ker_diff[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker_diff[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker_diff[2*Tin_C_l14];
PI_L2 float l15_ker_diff[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];
PI_L2 float l16_ker_diff[Tin_C_l16 * Tout_C_l16 * Tker_H_l16 * Tker_W_l16];
PI_L2 float l17_ker_diff[2*Tin_C_l17];
PI_L2 float l18_ker_diff[Tin_C_l18 * Tout_C_l18 * Tker_H_l18 * Tker_W_l18];
PI_L2 float l19_ker_diff[Tin_C_l19 * Tout_C_l19 * Tker_H_l19 * Tker_W_l19];
PI_L2 float l20_ker_diff[2*Tin_C_l20];
PI_L2 float l21_ker_diff[Tin_C_l21 * Tout_C_l21 * Tker_H_l21 * Tker_W_l21];
PI_L2 float l22_ker_diff[Tin_C_l22 * Tout_C_l22 * Tker_H_l22 * Tker_W_l22];

// Define I/O tensors
PI_L2 float l0_in[Tin_C_l0 * Tin_H_l0 * Tin_W_l0];
PI_L2 float l1_in[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L2 float l2_in[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L2 float l3_in[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L2 float l4_in[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L2 float l5_in[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L2 float l6_in[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L2 float l7_in[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L2 float l8_in[Tin_C_l8 * Tin_H_l8 * Tin_W_l8];
PI_L2 float l9_in[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L2 float l10_in[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L2 float l11_in[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L2 float l12_in[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L2 float l13_in[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L2 float l14_in[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L2 float l15_in[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L2 float l16_in[Tin_C_l16 * Tin_H_l16 * Tin_W_l16];
PI_L2 float l17_in[Tin_C_l17 * Tin_H_l17 * Tin_W_l17];
PI_L2 float l18_in[Tin_C_l18 * Tin_H_l18 * Tin_W_l18];
PI_L2 float l19_in[Tin_C_l19 * Tin_H_l19 * Tin_W_l19];
PI_L2 float l20_in[Tin_C_l20 * Tin_H_l20 * Tin_W_l20];
PI_L2 float l21_in[Tin_C_l21 * Tin_H_l21 * Tin_W_l21];
PI_L2 float l22_in[Tin_C_l22 * Tin_H_l22 * Tin_W_l22];
PI_L2 float l22_out[Tout_C_l22 * Tout_H_l22 * Tout_W_l22];

// Define IM2COL buffer for all the convolutions
PI_L1 float im2col_buffer[Tout_C_l0*Tker_H_l0*Tker_W_l0*Tin_H_l0*Tin_W_l0];

// Define transposition / block transposition buffer for all conv2d and PW layers
PI_L1 float bt_buffer[Tin_C_l19*Tout_C_l19*Tker_H_l19*Tker_W_l19];

// Define error propagation tensors
PI_L2 float l1_in_diff[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L2 float l2_in_diff[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L2 float l3_in_diff[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L2 float l4_in_diff[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L2 float l5_in_diff[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L2 float l6_in_diff[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L2 float l7_in_diff[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L2 float l8_in_diff[Tin_C_l8 * Tin_H_l8 * Tin_W_l8];
PI_L2 float l9_in_diff[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L2 float l10_in_diff[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L2 float l11_in_diff[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L2 float l12_in_diff[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L2 float l13_in_diff[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L2 float l14_in_diff[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L2 float l15_in_diff[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L2 float l16_in_diff[Tin_C_l16 * Tin_H_l16 * Tin_W_l16];
PI_L2 float l17_in_diff[Tin_C_l17 * Tin_H_l17 * Tin_W_l17];
PI_L2 float l18_in_diff[Tin_C_l18 * Tin_H_l18 * Tin_W_l18];
PI_L2 float l19_in_diff[Tin_C_l19 * Tin_H_l19 * Tin_W_l19];
PI_L2 float l20_in_diff[Tin_C_l20 * Tin_H_l20 * Tin_W_l20];
PI_L2 float l21_in_diff[Tin_C_l21 * Tin_H_l21 * Tin_W_l21];
PI_L2 float l22_in_diff[Tin_C_l22 * Tin_H_l22 * Tin_W_l22];
PI_L2 float l22_out_diff[Tout_C_l22 * Tout_H_l22 * Tout_W_l22];

// Loss function configuration structure
PI_L1 struct loss_args loss_args;



/**
 * DNN BACKEND FUNCTIONS
**/

// DNN initialization function
void DNN_init()
{

// Assign pointers in L1
d0_blob.data = BUFF;
d0_blob.diff = BUFF;
w0_blob.data = BUFF;
w0_blob.diff = BUFF;
d1_blob.data = BUFF + MAX_SIZE/2;
d1_blob.diff = BUFF + MAX_SIZE/2;
w1_blob.data = BUFF + MAX_SIZE/2;
w1_blob.diff = BUFF + MAX_SIZE/2;
reset_arguments();

  // Layer 0
  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)			l0_in[i] = INPUT[i];
  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)		l0_ker[i] = init_WGT_l0[i];
  // Layer 1
  for(int i=0; i<2*Tin_C_l1; i++)		l1_ker[i] = init_WGT_l1[i];
  // Layer 2
  for(int i=0; i<Tin_C_l2*Tout_C_l2*Tker_H_l2*Tker_W_l2; i++)		l2_ker[i] = init_WGT_l2[i];
  // Layer 3
  //   Pooling kernel (no parameters)
  // Layer 4
  for(int i=0; i<Tin_C_l4*Tout_C_l4*Tker_H_l4*Tker_W_l4; i++)		l4_ker[i] = init_WGT_l4[i];
  // Layer 5
  for(int i=0; i<2*Tin_C_l5; i++)		l5_ker[i] = init_WGT_l5[i];
  // Layer 6
  for(int i=0; i<Tin_C_l6*Tout_C_l6*Tker_H_l6*Tker_W_l6; i++)		l6_ker[i] = init_WGT_l6[i];
  // Layer 7
  for(int i=0; i<Tin_C_l7*Tout_C_l7*Tker_H_l7*Tker_W_l7; i++)		l7_ker[i] = init_WGT_l7[i];
  // Layer 8
  for(int i=0; i<2*Tin_C_l8; i++)		l8_ker[i] = init_WGT_l8[i];
  // Layer 9
  for(int i=0; i<Tin_C_l9*Tout_C_l9*Tker_H_l9*Tker_W_l9; i++)		l9_ker[i] = init_WGT_l9[i];
  // Layer 10
  for(int i=0; i<Tin_C_l10*Tout_C_l10*Tker_H_l10*Tker_W_l10; i++)		l10_ker[i] = init_WGT_l10[i];
  // Layer 11
  for(int i=0; i<2*Tin_C_l11; i++)		l11_ker[i] = init_WGT_l11[i];
  // Layer 12
  for(int i=0; i<Tin_C_l12*Tout_C_l12*Tker_H_l12*Tker_W_l12; i++)		l12_ker[i] = init_WGT_l12[i];
  // Layer 13
  for(int i=0; i<Tin_C_l13*Tout_C_l13*Tker_H_l13*Tker_W_l13; i++)		l13_ker[i] = init_WGT_l13[i];
  // Layer 14
  for(int i=0; i<2*Tin_C_l14; i++)		l14_ker[i] = init_WGT_l14[i];
  // Layer 15
  for(int i=0; i<Tin_C_l15*Tout_C_l15*Tker_H_l15*Tker_W_l15; i++)		l15_ker[i] = init_WGT_l15[i];
  // Layer 16
  for(int i=0; i<Tin_C_l16*Tout_C_l16*Tker_H_l16*Tker_W_l16; i++)		l16_ker[i] = init_WGT_l16[i];
  // Layer 17
  for(int i=0; i<2*Tin_C_l17; i++)		l17_ker[i] = init_WGT_l17[i];
  // Layer 18
  for(int i=0; i<Tin_C_l18*Tout_C_l18*Tker_H_l18*Tker_W_l18; i++)		l18_ker[i] = init_WGT_l18[i];
  // Layer 19
  for(int i=0; i<Tin_C_l19*Tout_C_l19*Tker_H_l19*Tker_W_l19; i++)		l19_ker[i] = init_WGT_l19[i];
  // Layer 20
  for(int i=0; i<2*Tin_C_l20; i++)		l20_ker[i] = init_WGT_l20[i];
  // Layer 21
  for(int i=0; i<Tin_C_l21*Tout_C_l21*Tker_H_l21*Tker_W_l21; i++)		l21_ker[i] = init_WGT_l21[i];
  // Layer 22
  for(int i=0; i<Tin_C_l22*Tout_C_l22*Tker_H_l22*Tker_W_l22; i++)		l22_ker[i] = init_WGT_l22[i];

  // Connect tensors to blobs


//Connecting conv2d
  // Layer 0
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;
  layer0_in.C = Tin_C_l0;
  layer0_in.H = Tin_H_l0;
  layer0_in.W = Tin_W_l0;
  layer0_wgt.data = l0_ker;
  layer0_wgt.diff = l0_ker_diff;
  layer0_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;
  layer0_wgt.C = Tin_C_l0;
  layer0_wgt.H = Tker_H_l0;
  layer0_wgt.W = Tker_W_l0;
  layer0_out.data = l1_in;
  layer0_out.diff = l1_in_diff;
  layer0_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;
  layer0_out.C = Tout_C_l0;
  layer0_out.H = Tout_H_l0;
  layer0_out.W = Tout_W_l0;


//Connecting InstNorm
  // Layer 1
  layer1_in.data = l1_in;
  layer1_in.diff = l1_in_diff;
  layer1_in.dim = Tin_C_l1*Tin_H_l1*Tin_W_l1;
  layer1_in.C = Tin_C_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.W = Tin_W_l1;
  layer1_wgt.data = l1_ker;
  layer1_wgt.diff = l1_ker_diff;
  layer1_wgt.dim = 2*Tin_C_l1;
  layer1_wgt.C = Tin_C_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_out.data = l2_in;
  layer1_out.diff = l2_in_diff;
  layer1_out.dim = Tout_C_l1*Tout_H_l1*Tout_W_l1;
  layer1_out.C = Tout_C_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.W = Tout_W_l1;


//Connecting ReLU
  // Layer 2
  layer2_in.data = l2_in;
  layer2_in.diff = l2_in_diff;
  layer2_in.dim = Tin_C_l2*Tin_H_l2*Tin_W_l2;
  layer2_in.C = Tin_C_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.W = Tin_W_l2;
  layer2_wgt.data = l2_ker;
  layer2_wgt.diff = l2_ker_diff;
  layer2_wgt.dim = Tin_C_l2*Tout_C_l2*Tker_H_l2*Tker_W_l2;
  layer2_wgt.C = Tin_C_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_out.data = l3_in;
  layer2_out.diff = l3_in_diff;
  layer2_out.dim = Tout_C_l2*Tout_H_l2*Tout_W_l2;
  layer2_out.C = Tout_C_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.W = Tout_W_l2;


//Connecting MaxPool
  // Layer 3
  layer3_in.data = l3_in;
  layer3_in.diff = l3_in_diff;
  layer3_in.dim = Tin_C_l3*Tin_H_l3*Tin_W_l3;
  layer3_in.C = Tin_C_l3;
  layer3_in.H = Tin_H_l3;
  layer3_in.W = Tin_W_l3;
  layer3_wgt.data = l3_ker;
  layer3_wgt.diff = l3_ker_diff;
  layer3_wgt.dim = Tin_C_l3*Tout_C_l3*Tker_H_l3*Tker_W_l3;
  layer3_wgt.C = Tin_C_l3;
  layer3_wgt.H = Tker_H_l3;
  layer3_wgt.W = Tker_W_l3;
  layer3_out.data = l4_in;
  layer3_out.diff = l4_in_diff;
  layer3_out.dim = Tout_C_l3*Tout_H_l3*Tout_W_l3;
  layer3_out.C = Tout_C_l3;
  layer3_out.H = Tout_H_l3;
  layer3_out.W = Tout_W_l3;


//Connecting conv2d
  // Layer 4
  layer4_in.data = l4_in;
  layer4_in.diff = l4_in_diff;
  layer4_in.dim = Tin_C_l4*Tin_H_l4*Tin_W_l4;
  layer4_in.C = Tin_C_l4;
  layer4_in.H = Tin_H_l4;
  layer4_in.W = Tin_W_l4;
  layer4_wgt.data = l4_ker;
  layer4_wgt.diff = l4_ker_diff;
  layer4_wgt.dim = Tin_C_l4*Tout_C_l4*Tker_H_l4*Tker_W_l4;
  layer4_wgt.C = Tin_C_l4;
  layer4_wgt.H = Tker_H_l4;
  layer4_wgt.W = Tker_W_l4;
  layer4_out.data = l5_in;
  layer4_out.diff = l5_in_diff;
  layer4_out.dim = Tout_C_l4*Tout_H_l4*Tout_W_l4;
  layer4_out.C = Tout_C_l4;
  layer4_out.H = Tout_H_l4;
  layer4_out.W = Tout_W_l4;


//Connecting InstNorm
  // Layer 5
  layer5_in.data = l5_in;
  layer5_in.diff = l5_in_diff;
  layer5_in.dim = Tin_C_l5*Tin_H_l5*Tin_W_l5;
  layer5_in.C = Tin_C_l5;
  layer5_in.H = Tin_H_l5;
  layer5_in.W = Tin_W_l5;
  layer5_wgt.data = l5_ker;
  layer5_wgt.diff = l5_ker_diff;
  layer5_wgt.dim = 2*Tin_C_l5;
  layer5_wgt.C = Tin_C_l5;
  layer5_wgt.H = Tker_H_l5;
  layer5_wgt.W = Tker_W_l5;
  layer5_out.data = l6_in;
  layer5_out.diff = l6_in_diff;
  layer5_out.dim = Tout_C_l5*Tout_H_l5*Tout_W_l5;
  layer5_out.C = Tout_C_l5;
  layer5_out.H = Tout_H_l5;
  layer5_out.W = Tout_W_l5;


//Connecting ReLU
  // Layer 6
  layer6_in.data = l6_in;
  layer6_in.diff = l6_in_diff;
  layer6_in.dim = Tin_C_l6*Tin_H_l6*Tin_W_l6;
  layer6_in.C = Tin_C_l6;
  layer6_in.H = Tin_H_l6;
  layer6_in.W = Tin_W_l6;
  layer6_wgt.data = l6_ker;
  layer6_wgt.diff = l6_ker_diff;
  layer6_wgt.dim = Tin_C_l6*Tout_C_l6*Tker_H_l6*Tker_W_l6;
  layer6_wgt.C = Tin_C_l6;
  layer6_wgt.H = Tker_H_l6;
  layer6_wgt.W = Tker_W_l6;
  layer6_out.data = l7_in;
  layer6_out.diff = l7_in_diff;
  layer6_out.dim = Tout_C_l6*Tout_H_l6*Tout_W_l6;
  layer6_out.C = Tout_C_l6;
  layer6_out.H = Tout_H_l6;
  layer6_out.W = Tout_W_l6;


//Connecting conv2d
  // Layer 7
  layer7_in.data = l7_in;
  layer7_in.diff = l7_in_diff;
  layer7_in.dim = Tin_C_l7*Tin_H_l7*Tin_W_l7;
  layer7_in.C = Tin_C_l7;
  layer7_in.H = Tin_H_l7;
  layer7_in.W = Tin_W_l7;
  layer7_wgt.data = l7_ker;
  layer7_wgt.diff = l7_ker_diff;
  layer7_wgt.dim = Tin_C_l7*Tout_C_l7*Tker_H_l7*Tker_W_l7;
  layer7_wgt.C = Tin_C_l7;
  layer7_wgt.H = Tker_H_l7;
  layer7_wgt.W = Tker_W_l7;
  layer7_out.data = l8_in;
  layer7_out.diff = l8_in_diff;
  layer7_out.dim = Tout_C_l7*Tout_H_l7*Tout_W_l7;
  layer7_out.C = Tout_C_l7;
  layer7_out.H = Tout_H_l7;
  layer7_out.W = Tout_W_l7;


//Connecting InstNorm
  // Layer 8
  layer8_in.data = l8_in;
  layer8_in.diff = l8_in_diff;
  layer8_in.dim = Tin_C_l8*Tin_H_l8*Tin_W_l8;
  layer8_in.C = Tin_C_l8;
  layer8_in.H = Tin_H_l8;
  layer8_in.W = Tin_W_l8;
  layer8_wgt.data = l8_ker;
  layer8_wgt.diff = l8_ker_diff;
  layer8_wgt.dim = 2*Tin_C_l8;
  layer8_wgt.C = Tin_C_l8;
  layer8_wgt.H = Tker_H_l8;
  layer8_wgt.W = Tker_W_l8;
  layer8_out.data = l9_in;
  layer8_out.diff = l9_in_diff;
  layer8_out.dim = Tout_C_l8*Tout_H_l8*Tout_W_l8;
  layer8_out.C = Tout_C_l8;
  layer8_out.H = Tout_H_l8;
  layer8_out.W = Tout_W_l8;


//Connecting ReLU
  // Layer 9
  layer9_in.data = l9_in;
  layer9_in.diff = l9_in_diff;
  layer9_in.dim = Tin_C_l9*Tin_H_l9*Tin_W_l9;
  layer9_in.C = Tin_C_l9;
  layer9_in.H = Tin_H_l9;
  layer9_in.W = Tin_W_l9;
  layer9_wgt.data = l9_ker;
  layer9_wgt.diff = l9_ker_diff;
  layer9_wgt.dim = Tin_C_l9*Tout_C_l9*Tker_H_l9*Tker_W_l9;
  layer9_wgt.C = Tin_C_l9;
  layer9_wgt.H = Tker_H_l9;
  layer9_wgt.W = Tker_W_l9;
  layer9_out.data = l10_in;
  layer9_out.diff = l10_in_diff;
  layer9_out.dim = Tout_C_l9*Tout_H_l9*Tout_W_l9;
  layer9_out.C = Tout_C_l9;
  layer9_out.H = Tout_H_l9;
  layer9_out.W = Tout_W_l9;


//Connecting conv2d
  // Layer 10
  layer10_in.data = l10_in;
  layer10_in.diff = l10_in_diff;
  layer10_in.dim = Tin_C_l10*Tin_H_l10*Tin_W_l10;
  layer10_in.C = Tin_C_l10;
  layer10_in.H = Tin_H_l10;
  layer10_in.W = Tin_W_l10;
  layer10_wgt.data = l10_ker;
  layer10_wgt.diff = l10_ker_diff;
  layer10_wgt.dim = Tin_C_l10*Tout_C_l10*Tker_H_l10*Tker_W_l10;
  layer10_wgt.C = Tin_C_l10;
  layer10_wgt.H = Tker_H_l10;
  layer10_wgt.W = Tker_W_l10;
  layer10_out.data = l11_in;
  layer10_out.diff = l11_in_diff;
  layer10_out.dim = Tout_C_l10*Tout_H_l10*Tout_W_l10;
  layer10_out.C = Tout_C_l10;
  layer10_out.H = Tout_H_l10;
  layer10_out.W = Tout_W_l10;


//Connecting InstNorm
  // Layer 11
  layer11_in.data = l11_in;
  layer11_in.diff = l11_in_diff;
  layer11_in.dim = Tin_C_l11*Tin_H_l11*Tin_W_l11;
  layer11_in.C = Tin_C_l11;
  layer11_in.H = Tin_H_l11;
  layer11_in.W = Tin_W_l11;
  layer11_wgt.data = l11_ker;
  layer11_wgt.diff = l11_ker_diff;
  layer11_wgt.dim = 2*Tin_C_l11;
  layer11_wgt.C = Tin_C_l11;
  layer11_wgt.H = Tker_H_l11;
  layer11_wgt.W = Tker_W_l11;
  layer11_out.data = l12_in;
  layer11_out.diff = l12_in_diff;
  layer11_out.dim = Tout_C_l11*Tout_H_l11*Tout_W_l11;
  layer11_out.C = Tout_C_l11;
  layer11_out.H = Tout_H_l11;
  layer11_out.W = Tout_W_l11;


//Connecting ReLU
  // Layer 12
  layer12_in.data = l12_in;
  layer12_in.diff = l12_in_diff;
  layer12_in.dim = Tin_C_l12*Tin_H_l12*Tin_W_l12;
  layer12_in.C = Tin_C_l12;
  layer12_in.H = Tin_H_l12;
  layer12_in.W = Tin_W_l12;
  layer12_wgt.data = l12_ker;
  layer12_wgt.diff = l12_ker_diff;
  layer12_wgt.dim = Tin_C_l12*Tout_C_l12*Tker_H_l12*Tker_W_l12;
  layer12_wgt.C = Tin_C_l12;
  layer12_wgt.H = Tker_H_l12;
  layer12_wgt.W = Tker_W_l12;
  layer12_out.data = l13_in;
  layer12_out.diff = l13_in_diff;
  layer12_out.dim = Tout_C_l12*Tout_H_l12*Tout_W_l12;
  layer12_out.C = Tout_C_l12;
  layer12_out.H = Tout_H_l12;
  layer12_out.W = Tout_W_l12;


//Connecting conv2d
  // Layer 13
  layer13_in.data = l13_in;
  layer13_in.diff = l13_in_diff;
  layer13_in.dim = Tin_C_l13*Tin_H_l13*Tin_W_l13;
  layer13_in.C = Tin_C_l13;
  layer13_in.H = Tin_H_l13;
  layer13_in.W = Tin_W_l13;
  layer13_wgt.data = l13_ker;
  layer13_wgt.diff = l13_ker_diff;
  layer13_wgt.dim = Tin_C_l13*Tout_C_l13*Tker_H_l13*Tker_W_l13;
  layer13_wgt.C = Tin_C_l13;
  layer13_wgt.H = Tker_H_l13;
  layer13_wgt.W = Tker_W_l13;
  layer13_out.data = l14_in;
  layer13_out.diff = l14_in_diff;
  layer13_out.dim = Tout_C_l13*Tout_H_l13*Tout_W_l13;
  layer13_out.C = Tout_C_l13;
  layer13_out.H = Tout_H_l13;
  layer13_out.W = Tout_W_l13;


//Connecting InstNorm
  // Layer 14
  layer14_in.data = l14_in;
  layer14_in.diff = l14_in_diff;
  layer14_in.dim = Tin_C_l14*Tin_H_l14*Tin_W_l14;
  layer14_in.C = Tin_C_l14;
  layer14_in.H = Tin_H_l14;
  layer14_in.W = Tin_W_l14;
  layer14_wgt.data = l14_ker;
  layer14_wgt.diff = l14_ker_diff;
  layer14_wgt.dim = 2*Tin_C_l14;
  layer14_wgt.C = Tin_C_l14;
  layer14_wgt.H = Tker_H_l14;
  layer14_wgt.W = Tker_W_l14;
  layer14_out.data = l15_in;
  layer14_out.diff = l15_in_diff;
  layer14_out.dim = Tout_C_l14*Tout_H_l14*Tout_W_l14;
  layer14_out.C = Tout_C_l14;
  layer14_out.H = Tout_H_l14;
  layer14_out.W = Tout_W_l14;


//Connecting ReLU
  // Layer 15
  layer15_in.data = l15_in;
  layer15_in.diff = l15_in_diff;
  layer15_in.dim = Tin_C_l15*Tin_H_l15*Tin_W_l15;
  layer15_in.C = Tin_C_l15;
  layer15_in.H = Tin_H_l15;
  layer15_in.W = Tin_W_l15;
  layer15_wgt.data = l15_ker;
  layer15_wgt.diff = l15_ker_diff;
  layer15_wgt.dim = Tin_C_l15*Tout_C_l15*Tker_H_l15*Tker_W_l15;
  layer15_wgt.C = Tin_C_l15;
  layer15_wgt.H = Tker_H_l15;
  layer15_wgt.W = Tker_W_l15;
  layer15_out.data = l16_in;
  layer15_out.diff = l16_in_diff;
  layer15_out.dim = Tout_C_l15*Tout_H_l15*Tout_W_l15;
  layer15_out.C = Tout_C_l15;
  layer15_out.H = Tout_H_l15;
  layer15_out.W = Tout_W_l15;


//Connecting conv2d
  // Layer 16
  layer16_in.data = l16_in;
  layer16_in.diff = l16_in_diff;
  layer16_in.dim = Tin_C_l16*Tin_H_l16*Tin_W_l16;
  layer16_in.C = Tin_C_l16;
  layer16_in.H = Tin_H_l16;
  layer16_in.W = Tin_W_l16;
  layer16_wgt.data = l16_ker;
  layer16_wgt.diff = l16_ker_diff;
  layer16_wgt.dim = Tin_C_l16*Tout_C_l16*Tker_H_l16*Tker_W_l16;
  layer16_wgt.C = Tin_C_l16;
  layer16_wgt.H = Tker_H_l16;
  layer16_wgt.W = Tker_W_l16;
  layer16_out.data = l17_in;
  layer16_out.diff = l17_in_diff;
  layer16_out.dim = Tout_C_l16*Tout_H_l16*Tout_W_l16;
  layer16_out.C = Tout_C_l16;
  layer16_out.H = Tout_H_l16;
  layer16_out.W = Tout_W_l16;


//Connecting InstNorm
  // Layer 17
  layer17_in.data = l17_in;
  layer17_in.diff = l17_in_diff;
  layer17_in.dim = Tin_C_l17*Tin_H_l17*Tin_W_l17;
  layer17_in.C = Tin_C_l17;
  layer17_in.H = Tin_H_l17;
  layer17_in.W = Tin_W_l17;
  layer17_wgt.data = l17_ker;
  layer17_wgt.diff = l17_ker_diff;
  layer17_wgt.dim = 2*Tin_C_l17;
  layer17_wgt.C = Tin_C_l17;
  layer17_wgt.H = Tker_H_l17;
  layer17_wgt.W = Tker_W_l17;
  layer17_out.data = l18_in;
  layer17_out.diff = l18_in_diff;
  layer17_out.dim = Tout_C_l17*Tout_H_l17*Tout_W_l17;
  layer17_out.C = Tout_C_l17;
  layer17_out.H = Tout_H_l17;
  layer17_out.W = Tout_W_l17;


//Connecting ReLU
  // Layer 18
  layer18_in.data = l18_in;
  layer18_in.diff = l18_in_diff;
  layer18_in.dim = Tin_C_l18*Tin_H_l18*Tin_W_l18;
  layer18_in.C = Tin_C_l18;
  layer18_in.H = Tin_H_l18;
  layer18_in.W = Tin_W_l18;
  layer18_wgt.data = l18_ker;
  layer18_wgt.diff = l18_ker_diff;
  layer18_wgt.dim = Tin_C_l18*Tout_C_l18*Tker_H_l18*Tker_W_l18;
  layer18_wgt.C = Tin_C_l18;
  layer18_wgt.H = Tker_H_l18;
  layer18_wgt.W = Tker_W_l18;
  layer18_out.data = l19_in;
  layer18_out.diff = l19_in_diff;
  layer18_out.dim = Tout_C_l18*Tout_H_l18*Tout_W_l18;
  layer18_out.C = Tout_C_l18;
  layer18_out.H = Tout_H_l18;
  layer18_out.W = Tout_W_l18;


//Connecting conv2d
  // Layer 19
  layer19_in.data = l19_in;
  layer19_in.diff = l19_in_diff;
  layer19_in.dim = Tin_C_l19*Tin_H_l19*Tin_W_l19;
  layer19_in.C = Tin_C_l19;
  layer19_in.H = Tin_H_l19;
  layer19_in.W = Tin_W_l19;
  layer19_wgt.data = l19_ker;
  layer19_wgt.diff = l19_ker_diff;
  layer19_wgt.dim = Tin_C_l19*Tout_C_l19*Tker_H_l19*Tker_W_l19;
  layer19_wgt.C = Tin_C_l19;
  layer19_wgt.H = Tker_H_l19;
  layer19_wgt.W = Tker_W_l19;
  layer19_out.data = l20_in;
  layer19_out.diff = l20_in_diff;
  layer19_out.dim = Tout_C_l19*Tout_H_l19*Tout_W_l19;
  layer19_out.C = Tout_C_l19;
  layer19_out.H = Tout_H_l19;
  layer19_out.W = Tout_W_l19;


//Connecting InstNorm
  // Layer 20
  layer20_in.data = l20_in;
  layer20_in.diff = l20_in_diff;
  layer20_in.dim = Tin_C_l20*Tin_H_l20*Tin_W_l20;
  layer20_in.C = Tin_C_l20;
  layer20_in.H = Tin_H_l20;
  layer20_in.W = Tin_W_l20;
  layer20_wgt.data = l20_ker;
  layer20_wgt.diff = l20_ker_diff;
  layer20_wgt.dim = 2*Tin_C_l20;
  layer20_wgt.C = Tin_C_l20;
  layer20_wgt.H = Tker_H_l20;
  layer20_wgt.W = Tker_W_l20;
  layer20_out.data = l21_in;
  layer20_out.diff = l21_in_diff;
  layer20_out.dim = Tout_C_l20*Tout_H_l20*Tout_W_l20;
  layer20_out.C = Tout_C_l20;
  layer20_out.H = Tout_H_l20;
  layer20_out.W = Tout_W_l20;


//Connecting ReLU
  // Layer 21
  layer21_in.data = l21_in;
  layer21_in.diff = l21_in_diff;
  layer21_in.dim = Tin_C_l21*Tin_H_l21*Tin_W_l21;
  layer21_in.C = Tin_C_l21;
  layer21_in.H = Tin_H_l21;
  layer21_in.W = Tin_W_l21;
  layer21_wgt.data = l21_ker;
  layer21_wgt.diff = l21_ker_diff;
  layer21_wgt.dim = Tin_C_l21*Tout_C_l21*Tker_H_l21*Tker_W_l21;
  layer21_wgt.C = Tin_C_l21;
  layer21_wgt.H = Tker_H_l21;
  layer21_wgt.W = Tker_W_l21;
  layer21_out.data = l22_in;
  layer21_out.diff = l22_in_diff;
  layer21_out.dim = Tout_C_l21*Tout_H_l21*Tout_W_l21;
  layer21_out.C = Tout_C_l21;
  layer21_out.H = Tout_H_l21;
  layer21_out.W = Tout_W_l21;


//Connecting linear
  // Layer 22
  layer22_in.data = l22_in;
  layer22_in.diff = l22_in_diff;
  layer22_in.dim = Tin_C_l22*Tin_H_l22*Tin_W_l22;
  layer22_in.C = Tin_C_l22;
  layer22_in.H = Tin_H_l22;
  layer22_in.W = Tin_W_l22;
  layer22_wgt.data = l22_ker;
  layer22_wgt.diff = l22_ker_diff;
  layer22_wgt.dim = Tin_C_l22*Tout_C_l22*Tker_H_l22*Tker_W_l22;
  layer22_wgt.C = Tin_C_l22;
  layer22_wgt.H = Tker_H_l22;
  layer22_wgt.W = Tker_W_l22;
  layer22_out.data = l22_out;
  layer22_out.diff = l22_out_diff;
  layer22_out.dim = Tout_C_l22*Tout_H_l22*Tout_W_l22;
  layer22_out.C = Tout_C_l22;
  layer22_out.H = Tout_H_l22;
  layer22_out.W = Tout_W_l22;

  // Configure layer structures
  // Layer 0
  l0_args.input = &in;
  l0_args.coeff = &wgt;
  l0_args.output = &out;
  l0_args.skip_in_grad = 1;
  l0_args.Lpad = 2;
  l0_args.Rpad = 2;
  l0_args.Upad = 2;
  l0_args.Dpad = 2;
  l0_args.stride_h = 2;
  l0_args.stride_w = 2;
  l0_args.i2c_buffer = (float*) im2col_buffer;
  l0_args.bt_buffer = (float*) bt_buffer;
  l0_args.HWC = 0;
  l0_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L0;
  l0_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L0;
  l0_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L0;
  l0_args.USE_IM2COL = 1;
  l0_args.USE_DMA_IM2COL = 0;
  // Layer 1
  l1_args.input = &in;
  l1_args.coeff = &wgt;
  l1_args.output = &out;
  l1_args.skip_in_grad = 0;
  // Layer 2
  l2_args.input = &in;
  l2_args.output = &out;
  // Layer 3
  //   Pooling layer (see next section)
  // Layer 4
  l4_args.input = &in;
  l4_args.coeff = &wgt;
  l4_args.output = &out;
  l4_args.skip_in_grad = 0;
  l4_args.Lpad = 1;
  l4_args.Rpad = 1;
  l4_args.Upad = 1;
  l4_args.Dpad = 1;
  l4_args.stride_h = 2;
  l4_args.stride_w = 2;
  l4_args.i2c_buffer = (float*) im2col_buffer;
  l4_args.bt_buffer = (float*) bt_buffer;
  l4_args.HWC = 0;
  l4_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L4;
  l4_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L4;
  l4_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L4;
  l4_args.USE_IM2COL = 1;
  l4_args.USE_DMA_IM2COL = 0;
  // Layer 5
  l5_args.input = &in;
  l5_args.coeff = &wgt;
  l5_args.output = &out;
  l5_args.skip_in_grad = 0;
  // Layer 6
  l6_args.input = &in;
  l6_args.output = &out;
  // Layer 7
  l7_args.input = &in;
  l7_args.coeff = &wgt;
  l7_args.output = &out;
  l7_args.skip_in_grad = 0;
  l7_args.Lpad = 1;
  l7_args.Rpad = 1;
  l7_args.Upad = 1;
  l7_args.Dpad = 1;
  l7_args.stride_h = 1;
  l7_args.stride_w = 1;
  l7_args.i2c_buffer = (float*) im2col_buffer;
  l7_args.bt_buffer = (float*) bt_buffer;
  l7_args.HWC = 0;
  l7_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L7;
  l7_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L7;
  l7_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L7;
  l7_args.USE_IM2COL = 1;
  l7_args.USE_DMA_IM2COL = 0;
  // Layer 8
  l8_args.input = &in;
  l8_args.coeff = &wgt;
  l8_args.output = &out;
  l8_args.skip_in_grad = 0;
  // Layer 9
  l9_args.input = &in;
  l9_args.output = &out;
  // Layer 10
  l10_args.input = &in;
  l10_args.coeff = &wgt;
  l10_args.output = &out;
  l10_args.skip_in_grad = 0;
  l10_args.Lpad = 1;
  l10_args.Rpad = 1;
  l10_args.Upad = 1;
  l10_args.Dpad = 1;
  l10_args.stride_h = 2;
  l10_args.stride_w = 2;
  l10_args.i2c_buffer = (float*) im2col_buffer;
  l10_args.bt_buffer = (float*) bt_buffer;
  l10_args.HWC = 0;
  l10_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L10;
  l10_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L10;
  l10_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L10;
  l10_args.USE_IM2COL = 1;
  l10_args.USE_DMA_IM2COL = 0;
  // Layer 11
  l11_args.input = &in;
  l11_args.coeff = &wgt;
  l11_args.output = &out;
  l11_args.skip_in_grad = 0;
  // Layer 12
  l12_args.input = &in;
  l12_args.output = &out;
  // Layer 13
  l13_args.input = &in;
  l13_args.coeff = &wgt;
  l13_args.output = &out;
  l13_args.skip_in_grad = 0;
  l13_args.Lpad = 1;
  l13_args.Rpad = 1;
  l13_args.Upad = 1;
  l13_args.Dpad = 1;
  l13_args.stride_h = 1;
  l13_args.stride_w = 1;
  l13_args.i2c_buffer = (float*) im2col_buffer;
  l13_args.bt_buffer = (float*) bt_buffer;
  l13_args.HWC = 0;
  l13_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L13;
  l13_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L13;
  l13_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L13;
  l13_args.USE_IM2COL = 1;
  l13_args.USE_DMA_IM2COL = 0;
  // Layer 14
  l14_args.input = &in;
  l14_args.coeff = &wgt;
  l14_args.output = &out;
  l14_args.skip_in_grad = 0;
  // Layer 15
  l15_args.input = &in;
  l15_args.output = &out;
  // Layer 16
  l16_args.input = &in;
  l16_args.coeff = &wgt;
  l16_args.output = &out;
  l16_args.skip_in_grad = 0;
  l16_args.Lpad = 1;
  l16_args.Rpad = 1;
  l16_args.Upad = 1;
  l16_args.Dpad = 1;
  l16_args.stride_h = 2;
  l16_args.stride_w = 2;
  l16_args.i2c_buffer = (float*) im2col_buffer;
  l16_args.bt_buffer = (float*) bt_buffer;
  l16_args.HWC = 0;
  l16_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L16;
  l16_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L16;
  l16_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L16;
  l16_args.USE_IM2COL = 1;
  l16_args.USE_DMA_IM2COL = 0;
  // Layer 17
  l17_args.input = &in;
  l17_args.coeff = &wgt;
  l17_args.output = &out;
  l17_args.skip_in_grad = 0;
  // Layer 18
  l18_args.input = &in;
  l18_args.output = &out;
  // Layer 19
  l19_args.input = &in;
  l19_args.coeff = &wgt;
  l19_args.output = &out;
  l19_args.skip_in_grad = 0;
  l19_args.Lpad = 1;
  l19_args.Rpad = 1;
  l19_args.Upad = 1;
  l19_args.Dpad = 1;
  l19_args.stride_h = 1;
  l19_args.stride_w = 1;
  l19_args.i2c_buffer = (float*) im2col_buffer;
  l19_args.bt_buffer = (float*) bt_buffer;
  l19_args.HWC = 0;
  l19_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L19;
  l19_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L19;
  l19_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L19;
  l19_args.USE_IM2COL = 1;
  l19_args.USE_DMA_IM2COL = 0;
  // Layer 20
  l20_args.input = &in;
  l20_args.coeff = &wgt;
  l20_args.output = &out;
  l20_args.skip_in_grad = 0;
  // Layer 21
  l21_args.input = &in;
  l21_args.output = &out;
  // Layer 22
  l22_args.input = &in;
  l22_args.coeff = &wgt;
  l22_args.output = &out;
  l22_args.skip_in_grad = 0;
  l22_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L22;
  l22_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L22;
  l22_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L22;

  // Connect blobs to pooling structures
  // Layer 3
  l3_pool_args.input = &layer3_in;
  l3_pool_args.output = &layer3_out;
  l3_pool_args.Hker = Tker_H_l3;
  l3_pool_args.Wker = Tker_W_l3;
  l3_pool_args.Hstride = Tstr_H_l3;
  l3_pool_args.Wstride = Tstr_W_l3;
}


// Forward pass function
void forward(){
	pi_cl_dma_flush();
	reset_arguments();

	get_dim(&layer0_in, &d0_blob);
	load((uint32_t) layer0_in.data, (uint32_t) d0_blob.data, 4*layer0_in.dim);
	get_dim(&layer0_wgt, &w0_blob);
	pi_cl_dma_cmd_wait(cmd_load);
	load((uint32_t) layer0_wgt.data, (uint32_t) w0_blob.data, 4*layer0_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_load);

	// Layer 0 (conv2d, 0, 1, 0)
	copy_struct_param((uint32_t) &l0_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer0_out, &d1_blob);
	get_dim(&layer1_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	load((uint32_t) layer1_wgt.data, (uint32_t) w1_blob.data, 4*layer1_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 1 (InstNorm, 1, 0, 1)
	copy_struct_param((uint32_t) &l1_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer1_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer0_out.data, 4*layer0_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 2 (ReLU, 0, 1, 0)
	get_dim(&layer2_out, &d1_blob);
	get_dim(&layer3_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer1_out.data, 4*layer1_out.dim);
	load((uint32_t) layer3_wgt.data, (uint32_t) w1_blob.data, 4*layer3_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 3 (MaxPool, 1, 0, 1)
	copy_struct_param((uint32_t) &l3_args, (uint32_t) &MaxPool_args, sizeof(MaxPool_args));
	get_dim(&layer3_out, &d0_blob);
	get_dim(&layer4_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer2_out.data, 4*layer2_out.dim);
	load((uint32_t) layer4_wgt.data, (uint32_t) w0_blob.data, 4*layer4_wgt.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l3_pool_args);

	// Layer 4 (conv2d, 0, 1, 0)
	copy_struct_param((uint32_t) &l4_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer4_out, &d1_blob);
	get_dim(&layer5_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer3_out.data, 4*layer3_out.dim);
	load((uint32_t) layer5_wgt.data, (uint32_t) w1_blob.data, 4*layer5_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 5 (InstNorm, 1, 0, 1)
	copy_struct_param((uint32_t) &l5_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer5_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer4_out.data, 4*layer4_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 6 (ReLU, 0, 1, 0)
	get_dim(&layer6_out, &d1_blob);
	get_dim(&layer7_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer5_out.data, 4*layer5_out.dim);
	load((uint32_t) layer7_wgt.data, (uint32_t) w1_blob.data, 4*layer7_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 7 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l7_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer7_out, &d0_blob);
	get_dim(&layer8_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer6_out.data, 4*layer6_out.dim);
	load((uint32_t) layer8_wgt.data, (uint32_t) w0_blob.data, 4*layer8_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 8 (InstNorm, 0, 1, 0)
	copy_struct_param((uint32_t) &l8_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer8_out, &d1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer7_out.data, 4*layer7_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 9 (ReLU, 1, 0, 1)
	get_dim(&layer9_out, &d0_blob);
	get_dim(&layer10_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer8_out.data, 4*layer8_out.dim);
	load((uint32_t) layer10_wgt.data, (uint32_t) w0_blob.data, 4*layer10_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 10 (conv2d, 0, 1, 0)
	copy_struct_param((uint32_t) &l10_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer10_out, &d1_blob);
	get_dim(&layer11_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer9_out.data, 4*layer9_out.dim);
	load((uint32_t) layer11_wgt.data, (uint32_t) w1_blob.data, 4*layer11_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 11 (InstNorm, 1, 0, 1)
	copy_struct_param((uint32_t) &l11_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer11_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer10_out.data, 4*layer10_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 12 (ReLU, 0, 1, 0)
	get_dim(&layer12_out, &d1_blob);
	get_dim(&layer13_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer11_out.data, 4*layer11_out.dim);
	load((uint32_t) layer13_wgt.data, (uint32_t) w1_blob.data, 4*layer13_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 13 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l13_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer13_out, &d0_blob);
	get_dim(&layer14_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer12_out.data, 4*layer12_out.dim);
	load((uint32_t) layer14_wgt.data, (uint32_t) w0_blob.data, 4*layer14_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 14 (InstNorm, 0, 1, 0)
	copy_struct_param((uint32_t) &l14_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer14_out, &d1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer13_out.data, 4*layer13_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 15 (ReLU, 1, 0, 1)
	get_dim(&layer15_out, &d0_blob);
	get_dim(&layer16_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer14_out.data, 4*layer14_out.dim);
	load((uint32_t) layer16_wgt.data, (uint32_t) w0_blob.data, 4*layer16_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 16 (conv2d, 0, 1, 0)
	copy_struct_param((uint32_t) &l16_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer16_out, &d1_blob);
	get_dim(&layer17_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer15_out.data, 4*layer15_out.dim);
	load((uint32_t) layer17_wgt.data, (uint32_t) w1_blob.data, 4*layer17_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 17 (InstNorm, 1, 0, 1)
	copy_struct_param((uint32_t) &l17_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer17_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer16_out.data, 4*layer16_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 18 (ReLU, 0, 1, 0)
	get_dim(&layer18_out, &d1_blob);
	get_dim(&layer19_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer17_out.data, 4*layer17_out.dim);
	load((uint32_t) layer19_wgt.data, (uint32_t) w1_blob.data, 4*layer19_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 19 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l19_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer19_out, &d0_blob);
	get_dim(&layer20_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer18_out.data, 4*layer18_out.dim);
	load((uint32_t) layer20_wgt.data, (uint32_t) w0_blob.data, 4*layer20_wgt.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 20 (InstNorm, 0, 1, 0)
	copy_struct_param((uint32_t) &l20_args, (uint32_t) &InstNorm_args, sizeof(InstNorm_args));
	get_dim(&layer20_out, &d1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer19_out.data, 4*layer19_out.dim);
	pulp_instnorm_fp32_fw_cl(&InstNorm_args);

	// Layer 21 (ReLU, 1, 0, 1)
	get_dim(&layer21_out, &d0_blob);
	get_dim(&layer22_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer20_out.data, 4*layer20_out.dim);
	load((uint32_t) layer22_wgt.data, (uint32_t) w0_blob.data, 4*layer22_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 22 (linear, 0, 1, 0)
	copy_struct_param((uint32_t) &l22_args, (uint32_t) &linear_args, sizeof(linear_args));
	get_dim(&layer22_out, &d1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer21_out.data, 4*layer21_out.dim);
	pulp_linear_fp32_fw_cl(&linear_args);

	store((uint32_t) out.data, (uint32_t) layer22_out.data, 4*layer22_out.dim);
	pi_cl_dma_cmd_wait(cmd_store);
}

// Backward pass function
void backward()
{

	// Layer 22 (linear, 1, 0, 1)
	copy_struct_param((unsigned int) &l22_args, (unsigned int) &linear_args, sizeof(l22_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer22_out.diff, 4*layer22_out.dim);
	pulp_linear_fp32_bw_param_grads_cl(&linear_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer22_wgt.diff, 4*layer22_wgt.dim);
	pulp_linear_fp32_bw_input_grads_cl(&linear_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer21_in, &d1_blob);
	load((uint32_t) layer21_in.data, (uint32_t) d1_blob.data, 4*layer21_in.dim);
	pi_cl_dma_flush();

	// Layer 21 (ReLU, 0, 1, 0)
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer21_out.diff, 4*layer21_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer20_in, &d0_blob);
	load((uint32_t) layer20_in.data, (uint32_t) d0_blob.data, 4*layer20_in.dim);
	get_dim( &layer20_wgt, &w0_blob);
	load((uint32_t) layer20_wgt.data, (uint32_t) w0_blob.data, 4*layer20_wgt.dim);
	pi_cl_dma_flush();

	// Layer 20 (InstNorm, 1, 0, 1)
	copy_struct_param((unsigned int) &l20_args, (unsigned int) &InstNorm_args, sizeof(l20_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer20_out.diff, 4*layer20_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer20_wgt.diff, 4*layer20_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer19_in, &d1_blob);
	load((uint32_t) layer19_in.data, (uint32_t) d1_blob.data, 4*layer19_in.dim);
	get_dim( &layer19_wgt, &w1_blob);
	load((uint32_t) layer19_wgt.data, (uint32_t) w1_blob.data, 4*layer19_wgt.dim);
	pi_cl_dma_flush();

	// Layer 19 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l19_args, (unsigned int) &conv2d_args, sizeof(l19_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer19_out.diff, 4*layer19_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer19_wgt.diff, 4*layer19_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer18_in, &d0_blob);
	load((uint32_t) layer18_in.data, (uint32_t) d0_blob.data, 4*layer18_in.dim);
	pi_cl_dma_flush();

	// Layer 18 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer18_out.diff, 4*layer18_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer17_in, &d1_blob);
	load((uint32_t) layer17_in.data, (uint32_t) d1_blob.data, 4*layer17_in.dim);
	get_dim( &layer17_wgt, &w1_blob);
	load((uint32_t) layer17_wgt.data, (uint32_t) w1_blob.data, 4*layer17_wgt.dim);
	pi_cl_dma_flush();

	// Layer 17 (InstNorm, 0, 1, 0)
	copy_struct_param((unsigned int) &l17_args, (unsigned int) &InstNorm_args, sizeof(l17_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer17_out.diff, 4*layer17_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer17_wgt.diff, 4*layer17_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer16_in, &d0_blob);
	load((uint32_t) layer16_in.data, (uint32_t) d0_blob.data, 4*layer16_in.dim);
	get_dim( &layer16_wgt, &w0_blob);
	load((uint32_t) layer16_wgt.data, (uint32_t) w0_blob.data, 4*layer16_wgt.dim);
	pi_cl_dma_flush();

	// Layer 16 (conv2d, 1, 0, 1)
	copy_struct_param((unsigned int) &l16_args, (unsigned int) &conv2d_args, sizeof(l16_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer16_out.diff, 4*layer16_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer16_wgt.diff, 4*layer16_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer15_in, &d1_blob);
	load((uint32_t) layer15_in.data, (uint32_t) d1_blob.data, 4*layer15_in.dim);
	pi_cl_dma_flush();

	// Layer 15 (ReLU, 0, 1, 0)
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer15_out.diff, 4*layer15_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer14_in, &d0_blob);
	load((uint32_t) layer14_in.data, (uint32_t) d0_blob.data, 4*layer14_in.dim);
	get_dim( &layer14_wgt, &w0_blob);
	load((uint32_t) layer14_wgt.data, (uint32_t) w0_blob.data, 4*layer14_wgt.dim);
	pi_cl_dma_flush();

	// Layer 14 (InstNorm, 1, 0, 1)
	copy_struct_param((unsigned int) &l14_args, (unsigned int) &InstNorm_args, sizeof(l14_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer14_out.diff, 4*layer14_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer14_wgt.diff, 4*layer14_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer13_in, &d1_blob);
	load((uint32_t) layer13_in.data, (uint32_t) d1_blob.data, 4*layer13_in.dim);
	get_dim( &layer13_wgt, &w1_blob);
	load((uint32_t) layer13_wgt.data, (uint32_t) w1_blob.data, 4*layer13_wgt.dim);
	pi_cl_dma_flush();

	// Layer 13 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l13_args, (unsigned int) &conv2d_args, sizeof(l13_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer13_out.diff, 4*layer13_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer13_wgt.diff, 4*layer13_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer12_in, &d0_blob);
	load((uint32_t) layer12_in.data, (uint32_t) d0_blob.data, 4*layer12_in.dim);
	pi_cl_dma_flush();

	// Layer 12 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer12_out.diff, 4*layer12_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer11_in, &d1_blob);
	load((uint32_t) layer11_in.data, (uint32_t) d1_blob.data, 4*layer11_in.dim);
	get_dim( &layer11_wgt, &w1_blob);
	load((uint32_t) layer11_wgt.data, (uint32_t) w1_blob.data, 4*layer11_wgt.dim);
	pi_cl_dma_flush();

	// Layer 11 (InstNorm, 0, 1, 0)
	copy_struct_param((unsigned int) &l11_args, (unsigned int) &InstNorm_args, sizeof(l11_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer11_out.diff, 4*layer11_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer11_wgt.diff, 4*layer11_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer10_in, &d0_blob);
	load((uint32_t) layer10_in.data, (uint32_t) d0_blob.data, 4*layer10_in.dim);
	get_dim( &layer10_wgt, &w0_blob);
	load((uint32_t) layer10_wgt.data, (uint32_t) w0_blob.data, 4*layer10_wgt.dim);
	pi_cl_dma_flush();

	// Layer 10 (conv2d, 1, 0, 1)
	copy_struct_param((unsigned int) &l10_args, (unsigned int) &conv2d_args, sizeof(l10_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer10_out.diff, 4*layer10_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer10_wgt.diff, 4*layer10_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer9_in, &d1_blob);
	load((uint32_t) layer9_in.data, (uint32_t) d1_blob.data, 4*layer9_in.dim);
	pi_cl_dma_flush();

	// Layer 9 (ReLU, 0, 1, 0)
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer9_out.diff, 4*layer9_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer8_in, &d0_blob);
	load((uint32_t) layer8_in.data, (uint32_t) d0_blob.data, 4*layer8_in.dim);
	get_dim( &layer8_wgt, &w0_blob);
	load((uint32_t) layer8_wgt.data, (uint32_t) w0_blob.data, 4*layer8_wgt.dim);
	pi_cl_dma_flush();

	// Layer 8 (InstNorm, 1, 0, 1)
	copy_struct_param((unsigned int) &l8_args, (unsigned int) &InstNorm_args, sizeof(l8_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer8_out.diff, 4*layer8_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer8_wgt.diff, 4*layer8_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer7_in, &d1_blob);
	load((uint32_t) layer7_in.data, (uint32_t) d1_blob.data, 4*layer7_in.dim);
	get_dim( &layer7_wgt, &w1_blob);
	load((uint32_t) layer7_wgt.data, (uint32_t) w1_blob.data, 4*layer7_wgt.dim);
	pi_cl_dma_flush();

	// Layer 7 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l7_args, (unsigned int) &conv2d_args, sizeof(l7_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer7_out.diff, 4*layer7_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer7_wgt.diff, 4*layer7_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer6_in, &d0_blob);
	load((uint32_t) layer6_in.data, (uint32_t) d0_blob.data, 4*layer6_in.dim);
	pi_cl_dma_flush();

	// Layer 6 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer6_out.diff, 4*layer6_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer5_in, &d1_blob);
	load((uint32_t) layer5_in.data, (uint32_t) d1_blob.data, 4*layer5_in.dim);
	get_dim( &layer5_wgt, &w1_blob);
	load((uint32_t) layer5_wgt.data, (uint32_t) w1_blob.data, 4*layer5_wgt.dim);
	pi_cl_dma_flush();

	// Layer 5 (InstNorm, 0, 1, 0)
	copy_struct_param((unsigned int) &l5_args, (unsigned int) &InstNorm_args, sizeof(l5_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer5_out.diff, 4*layer5_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer5_wgt.diff, 4*layer5_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer4_in, &d0_blob);
	load((uint32_t) layer4_in.data, (uint32_t) d0_blob.data, 4*layer4_in.dim);
	get_dim( &layer4_wgt, &w0_blob);
	load((uint32_t) layer4_wgt.data, (uint32_t) w0_blob.data, 4*layer4_wgt.dim);
	pi_cl_dma_flush();

	// Layer 4 (conv2d, 1, 0, 1)
	copy_struct_param((unsigned int) &l4_args, (unsigned int) &conv2d_args, sizeof(l4_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer4_out.diff, 4*layer4_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer4_wgt.diff, 4*layer4_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer3_in, &d1_blob);
	load((uint32_t) layer3_in.data, (uint32_t) d1_blob.data, 4*layer3_in.dim);
	get_dim( &layer3_wgt, &w1_blob);
	load((uint32_t) layer3_wgt.data, (uint32_t) w1_blob.data, 4*layer3_wgt.dim);
	pi_cl_dma_flush();

	// Layer 3 (MaxPool, 0, 1, 0)
	copy_struct_param((unsigned int) &l3_args, (unsigned int) &MaxPool_args, sizeof(l3_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer3_out.diff, 4*layer3_out.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l3_pool_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer3_wgt.diff, 4*layer3_wgt.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l3_pool_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer2_in, &d0_blob);
	load((uint32_t) layer2_in.data, (uint32_t) d0_blob.data, 4*layer2_in.dim);
	pi_cl_dma_flush();

	// Layer 2 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer2_out.diff, 4*layer2_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer1_in, &d1_blob);
	load((uint32_t) layer1_in.data, (uint32_t) d1_blob.data, 4*layer1_in.dim);
	get_dim( &layer1_wgt, &w1_blob);
	load((uint32_t) layer1_wgt.data, (uint32_t) w1_blob.data, 4*layer1_wgt.dim);
	pi_cl_dma_flush();

	// Layer 1 (InstNorm, 0, 1, 0)
	copy_struct_param((unsigned int) &l1_args, (unsigned int) &InstNorm_args, sizeof(l1_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer1_out.diff, 4*layer1_out.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer1_wgt.diff, 4*layer1_wgt.dim);
	pulp_instnorm_fp32_bw_cl(&InstNorm_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer0_in, &d0_blob);
	load((uint32_t) layer0_in.data, (uint32_t) d0_blob.data, 4*layer0_in.dim);
	get_dim( &layer0_wgt, &w0_blob);
	load((uint32_t) layer0_wgt.data, (uint32_t) w0_blob.data, 4*layer0_wgt.dim);
	pi_cl_dma_flush();

	// Layer 0 (conv2d, 1, 0, 1)
	copy_struct_param((unsigned int) &l0_args, (unsigned int) &conv2d_args, sizeof(l0_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer0_out.diff, 4*layer0_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer0_wgt.diff, 4*layer0_wgt.dim);
}

// Compute loss and output gradient
void compute_loss()
{
	loss_args.output = &out;
	loss_args.target = out.diff;
	loss_args.wr_loss = &loss;
	load((uint32_t) LABEL, (uint32_t) out.diff, 4*OUT_SIZE);
	pi_cl_dma_cmd_wait(cmd_load);
  pulp_MSELoss(&loss_args);
	store((uint32_t) out.diff, (uint32_t) layer22_out.diff, 4*OUT_SIZE);
}

// Function to update the network
void update_weights()
{
	// Creates all structures needed
	struct optim_args opt_l0;
	struct optim_args opt_l1;
	struct optim_args opt_l4;
	struct optim_args opt_l5;
	struct optim_args opt_l7;
	struct optim_args opt_l8;
	struct optim_args opt_l10;
	struct optim_args opt_l11;
	struct optim_args opt_l13;
	struct optim_args opt_l14;
	struct optim_args opt_l16;
	struct optim_args opt_l17;
	struct optim_args opt_l19;
	struct optim_args opt_l20;
	struct optim_args opt_l22;

	pi_cl_dma_flush();
	get_dim(&layer0_wgt, &d0_blob);
	opt_l0.weights = &d0_blob;
	opt_l0.learning_rate = LEARNING_RATE;
	load((uint32_t) layer0_wgt.data, (uint32_t) d0_blob.data, 4*layer0_wgt.dim);
	load((uint32_t) layer0_wgt.diff, (uint32_t) d0_blob.diff, 4*layer0_wgt.dim);

	pi_cl_dma_flush();
	get_dim(&layer1_wgt, &d1_blob);
	opt_l1.weights = &d1_blob;
	opt_l1.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer1_wgt.data, (uint32_t) d1_blob.data, 4*layer1_wgt.dim);
	load((uint32_t) layer1_wgt.diff, (uint32_t) d1_blob.diff, 4*layer1_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
	store((uint32_t) d0_blob.data, (uint32_t) layer0_wgt.data, 4*layer0_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer4_wgt, &d0_blob);
	opt_l4.weights = &d0_blob;
	opt_l4.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer4_wgt.data, (uint32_t) d0_blob.data, 4*layer4_wgt.dim);
	load((uint32_t) layer4_wgt.diff, (uint32_t) d0_blob.diff, 4*layer4_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l1);
	store((uint32_t) d1_blob.data, (uint32_t) layer1_wgt.data, 4*layer1_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer5_wgt, &d1_blob);
	opt_l5.weights = &d1_blob;
	opt_l5.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer5_wgt.data, (uint32_t) d1_blob.data, 4*layer5_wgt.dim);
	load((uint32_t) layer5_wgt.diff, (uint32_t) d1_blob.diff, 4*layer5_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l4);
	store((uint32_t) d0_blob.data, (uint32_t) layer4_wgt.data, 4*layer4_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer7_wgt, &d0_blob);
	opt_l7.weights = &d0_blob;
	opt_l7.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer7_wgt.data, (uint32_t) d0_blob.data, 4*layer7_wgt.dim);
	load((uint32_t) layer7_wgt.diff, (uint32_t) d0_blob.diff, 4*layer7_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l5);
	store((uint32_t) d1_blob.data, (uint32_t) layer5_wgt.data, 4*layer5_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer8_wgt, &d1_blob);
	opt_l8.weights = &d1_blob;
	opt_l8.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer8_wgt.data, (uint32_t) d1_blob.data, 4*layer8_wgt.dim);
	load((uint32_t) layer8_wgt.diff, (uint32_t) d1_blob.diff, 4*layer8_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l7);
	store((uint32_t) d0_blob.data, (uint32_t) layer7_wgt.data, 4*layer7_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer10_wgt, &d0_blob);
	opt_l10.weights = &d0_blob;
	opt_l10.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer10_wgt.data, (uint32_t) d0_blob.data, 4*layer10_wgt.dim);
	load((uint32_t) layer10_wgt.diff, (uint32_t) d0_blob.diff, 4*layer10_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l8);
	store((uint32_t) d1_blob.data, (uint32_t) layer8_wgt.data, 4*layer8_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer11_wgt, &d1_blob);
	opt_l11.weights = &d1_blob;
	opt_l11.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer11_wgt.data, (uint32_t) d1_blob.data, 4*layer11_wgt.dim);
	load((uint32_t) layer11_wgt.diff, (uint32_t) d1_blob.diff, 4*layer11_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l10);
	store((uint32_t) d0_blob.data, (uint32_t) layer10_wgt.data, 4*layer10_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer13_wgt, &d0_blob);
	opt_l13.weights = &d0_blob;
	opt_l13.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer13_wgt.data, (uint32_t) d0_blob.data, 4*layer13_wgt.dim);
	load((uint32_t) layer13_wgt.diff, (uint32_t) d0_blob.diff, 4*layer13_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l11);
	store((uint32_t) d1_blob.data, (uint32_t) layer11_wgt.data, 4*layer11_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer14_wgt, &d1_blob);
	opt_l14.weights = &d1_blob;
	opt_l14.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer14_wgt.data, (uint32_t) d1_blob.data, 4*layer14_wgt.dim);
	load((uint32_t) layer14_wgt.diff, (uint32_t) d1_blob.diff, 4*layer14_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l13);
	store((uint32_t) d0_blob.data, (uint32_t) layer13_wgt.data, 4*layer13_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer16_wgt, &d0_blob);
	opt_l16.weights = &d0_blob;
	opt_l16.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer16_wgt.data, (uint32_t) d0_blob.data, 4*layer16_wgt.dim);
	load((uint32_t) layer16_wgt.diff, (uint32_t) d0_blob.diff, 4*layer16_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l14);
	store((uint32_t) d1_blob.data, (uint32_t) layer14_wgt.data, 4*layer14_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer17_wgt, &d1_blob);
	opt_l17.weights = &d1_blob;
	opt_l17.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer17_wgt.data, (uint32_t) d1_blob.data, 4*layer17_wgt.dim);
	load((uint32_t) layer17_wgt.diff, (uint32_t) d1_blob.diff, 4*layer17_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l16);
	store((uint32_t) d0_blob.data, (uint32_t) layer16_wgt.data, 4*layer16_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer19_wgt, &d0_blob);
	opt_l19.weights = &d0_blob;
	opt_l19.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer19_wgt.data, (uint32_t) d0_blob.data, 4*layer19_wgt.dim);
	load((uint32_t) layer19_wgt.diff, (uint32_t) d0_blob.diff, 4*layer19_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l17);
	store((uint32_t) d1_blob.data, (uint32_t) layer17_wgt.data, 4*layer17_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer20_wgt, &d1_blob);
	opt_l20.weights = &d1_blob;
	opt_l20.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer20_wgt.data, (uint32_t) d1_blob.data, 4*layer20_wgt.dim);
	load((uint32_t) layer20_wgt.diff, (uint32_t) d1_blob.diff, 4*layer20_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l19);
	store((uint32_t) d0_blob.data, (uint32_t) layer19_wgt.data, 4*layer19_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer22_wgt, &d0_blob);
	opt_l22.weights = &d0_blob;
	opt_l22.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer22_wgt.data, (uint32_t) d0_blob.data, 4*layer22_wgt.dim);
	load((uint32_t) layer22_wgt.diff, (uint32_t) d0_blob.diff, 4*layer22_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l20);
	store((uint32_t) d1_blob.data, (uint32_t) layer20_wgt.data, 4*layer20_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l22);
	store((uint32_t) d0_blob.data, (uint32_t) layer22_wgt.data, 4*layer22_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);
}



/**
 * DATA VISUALIZATION AND CHECK TOOLS
**/

// Function to print FW output
void print_output()
{
  printf("\nLayer 22 output:\n");

  for (int i=0; i<Tout_C_l22*Tout_H_l22*Tout_W_l22; i++)
  {
    printf("%f ", l22_out[i]);
    // Newline when an output row ends
    // if(!(i%Tout_W_l22)) printf("\n");
    // Newline when an output channel ends
    if(!(i%Tout_W_l22*Tout_H_l22)) printf("\n");
  }
}

// Function to check post-training output wrt Golden Model (GM)
void check_post_training_output()
{
  int integrity_check = 0;
  integrity_check = verify_tensor(l22_out, REFERENCE_OUTPUT, Tout_C_l22*Tout_H_l22*Tout_W_l22, TOLERANCE);
  if (integrity_check > 0)
    printf("\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\n");
}



/**
 * DNN MODEL TRAINING
**/

// Call for a complete training step
void net_step()
{
  printf("Initializing network..\n");
  DNN_init();
  printf("Testing DNN initialization forward..");
  forward();
  print_output();

  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  START_STATS();
  #endif

  for (int epoch=0; epoch<EPOCHS; epoch++)
  {
    forward();
    compute_loss();
    backward();
    update_weights();
  }

  #ifdef PROF_NET
  STOP_STATS();
  #endif

  // Check and print updated output
  forward();
  printf("Checking updated output..\n");
  check_post_training_output();
  print_output();
}

void load(uint32_t src, uint32_t dst, int dim){
	pi_cl_dma_cmd(src, dst, dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);}

void store(uint32_t src, uint32_t dst, int dim){
	pi_cl_dma_cmd(dst, src, dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);}

void get_dim(void * src_blob, void * dst_blob){
	struct blob * s = (struct blob * ) src_blob;
	struct blob * d = (struct blob * ) dst_blob;
	d->dim = s->dim;
	d->C = s->C;
	d->H = s->H;
	d->W = s->W;
	update();}

void update(){
	d0_blob.data = BUFF;
	d0_blob.diff = BUFF + d0_blob.dim;
	w0_blob.data = BUFF + 2*d0_blob.dim;
	w0_blob.diff = BUFF + 2*d0_blob.dim + w0_blob.dim;
	d1_blob.data = BUFF + MAX_SIZE/2;
	d1_blob.diff = BUFF + MAX_SIZE/2 + d1_blob.dim;
	w1_blob.data = BUFF + MAX_SIZE/2 + 2*d1_blob.dim;
	w1_blob.diff = BUFF + MAX_SIZE/2 + 2*d1_blob.dim + w1_blob.dim;}

void reset_arguments(){
	d0_blob.dim = 0;
	w0_blob.dim = 0;
	d1_blob.dim = 0;
	w1_blob.dim = 0;
	linear_args.output = &out;
	linear_args.input = &in;
	linear_args.coeff = &wgt;
	conv2d_args.output = &out;
	conv2d_args.input = &in;
	conv2d_args.coeff = &wgt;
	PW_args.output = &out;
	PW_args.input = &in;
	PW_args.coeff = &wgt;
	DW_args.output = &out;
	DW_args.input = &in;
	DW_args.coeff = &wgt;
	act_args.output = &out;
	act_args.input = &in;
	resconn_args.output = &out;
	resconn_args.lout = &in;
	resconn_args.skip = &wgt;
}


void copy_struct_param(unsigned int from, unsigned int to, int size){
	pi_cl_dma_cmd(from, to, size, PI_CL_DMA_DIR_EXT2LOC , cmd_struct);}
