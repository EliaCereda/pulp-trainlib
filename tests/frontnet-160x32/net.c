/**
 * INCLUDES
**/

#include "pulp_train.h"
#include "net.h"
#include "stats.h"

#include "init-defines.h"
#include "io_data.h"



/**
 * DATA
**/

// Define loss
PI_L1 float loss = 0;

// Define DNN blobs
PI_L1 struct blob layer0_in, layer0_wgt, layer0_out;
PI_L1 struct blob layer1_in, layer1_wgt, layer1_out;
PI_L1 struct blob layer2_in, layer2_wgt, layer2_out;
PI_L1 struct blob layer3_in, layer3_wgt, layer3_out;
PI_L1 struct blob layer4_in, layer4_wgt, layer4_out;
PI_L1 struct blob layer5_in, layer5_wgt, layer5_out;
PI_L1 struct blob layer6_in, layer6_wgt, layer6_out;
PI_L1 struct blob layer7_in, layer7_wgt, layer7_out;
PI_L1 struct blob layer8_in, layer8_wgt, layer8_out;
PI_L1 struct blob layer9_in, layer9_wgt, layer9_out;
PI_L1 struct blob layer10_in, layer10_wgt, layer10_out;
PI_L1 struct blob layer11_in, layer11_wgt, layer11_out;
PI_L1 struct blob layer12_in, layer12_wgt, layer12_out;
PI_L1 struct blob layer13_in, layer13_wgt, layer13_out;
PI_L1 struct blob layer14_in, layer14_wgt, layer14_out;
PI_L1 struct blob layer15_in, layer15_wgt, layer15_out;

// Define DNN layer structures
PI_L1 struct vect_sum_args vect_sum_args;
PI_L1 struct vect_sum_args_fp16 vect_sum_args_fp16;
PI_L1 struct Conv2D_args l0_args;
PI_L1 struct act_args l1_args;
PI_L1 struct Conv2D_args l3_args;
PI_L1 struct act_args l4_args;
PI_L1 struct Conv2D_args l5_args;
PI_L1 struct act_args l6_args;
PI_L1 struct Conv2D_args l7_args;
PI_L1 struct act_args l8_args;
PI_L1 struct Conv2D_args l9_args;
PI_L1 struct act_args l10_args;
PI_L1 struct Conv2D_args l11_args;
PI_L1 struct act_args l12_args;
PI_L1 struct Conv2D_args l13_args;
PI_L1 struct act_args l14_args;
PI_L1 struct Linear_args l15_args;

// Define Pooling Structures
PI_L1 struct pool_args l2_pool_args;

// Define kernel tensors
PI_L1 float l0_ker[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L1 float l1_ker[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L1 float l2_ker[1];
PI_L1 float l3_ker[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L1 float l4_ker[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L1 float l5_ker[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L1 float l6_ker[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L1 float l7_ker[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L1 float l8_ker[Tin_C_l8 * Tout_C_l8 * Tker_H_l8 * Tker_W_l8];
PI_L1 float l9_ker[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L1 float l10_ker[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L1 float l11_ker[Tin_C_l11 * Tout_C_l11 * Tker_H_l11 * Tker_W_l11];
PI_L1 float l12_ker[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L1 float l13_ker[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L1 float l14_ker[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L1 float l15_ker[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];

// Define kernel grad tensors
PI_L1 float l0_ker_diff[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L1 float l1_ker_diff[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L1 float l2_ker_diff[1];
PI_L1 float l3_ker_diff[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L1 float l4_ker_diff[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L1 float l5_ker_diff[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L1 float l6_ker_diff[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L1 float l7_ker_diff[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L1 float l8_ker_diff[Tin_C_l8 * Tout_C_l8 * Tker_H_l8 * Tker_W_l8];
PI_L1 float l9_ker_diff[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L1 float l10_ker_diff[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L1 float l11_ker_diff[Tin_C_l11 * Tout_C_l11 * Tker_H_l11 * Tker_W_l11];
PI_L1 float l12_ker_diff[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L1 float l13_ker_diff[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L1 float l14_ker_diff[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L1 float l15_ker_diff[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];

// Define I/O tensors
PI_L1 float l0_in[Tin_C_l0 * Tin_H_l0 * Tin_W_l0];
PI_L1 float l1_in[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 float l2_in[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l3_in[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L1 float l4_in[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L1 float l5_in[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L1 float l6_in[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L1 float l7_in[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L1 float l8_in[Tin_C_l8 * Tin_H_l8 * Tin_W_l8];
PI_L1 float l9_in[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L1 float l10_in[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L1 float l11_in[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L1 float l12_in[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L1 float l13_in[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L1 float l14_in[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L1 float l15_in[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L1 float l15_out[Tout_C_l15 * Tout_H_l15 * Tout_W_l15];

// Define IM2COL buffer for all the convolutions
PI_L1 float im2col_buffer[Tout_C_l0*Tker_H_l0*Tker_W_l0*Tin_H_l0*Tin_W_l0];

// Define transposition / block transposition buffer for all conv2d and PW layers
PI_L1 float bt_buffer[Tin_C_l13*Tout_C_l13*Tker_H_l13*Tker_W_l13];

// Define error propagation tensors
PI_L1 float l1_in_diff[Tin_C_l1 * Tin_H_l1 * Tin_W_l1];
PI_L1 float l2_in_diff[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l3_in_diff[Tin_C_l3 * Tin_H_l3 * Tin_W_l3];
PI_L1 float l4_in_diff[Tin_C_l4 * Tin_H_l4 * Tin_W_l4];
PI_L1 float l5_in_diff[Tin_C_l5 * Tin_H_l5 * Tin_W_l5];
PI_L1 float l6_in_diff[Tin_C_l6 * Tin_H_l6 * Tin_W_l6];
PI_L1 float l7_in_diff[Tin_C_l7 * Tin_H_l7 * Tin_W_l7];
PI_L1 float l8_in_diff[Tin_C_l8 * Tin_H_l8 * Tin_W_l8];
PI_L1 float l9_in_diff[Tin_C_l9 * Tin_H_l9 * Tin_W_l9];
PI_L1 float l10_in_diff[Tin_C_l10 * Tin_H_l10 * Tin_W_l10];
PI_L1 float l11_in_diff[Tin_C_l11 * Tin_H_l11 * Tin_W_l11];
PI_L1 float l12_in_diff[Tin_C_l12 * Tin_H_l12 * Tin_W_l12];
PI_L1 float l13_in_diff[Tin_C_l13 * Tin_H_l13 * Tin_W_l13];
PI_L1 float l14_in_diff[Tin_C_l14 * Tin_H_l14 * Tin_W_l14];
PI_L1 float l15_in_diff[Tin_C_l15 * Tin_H_l15 * Tin_W_l15];
PI_L1 float l15_out_diff[Tout_C_l15 * Tout_H_l15 * Tout_W_l15];

// Loss function configuration structure
PI_L1 struct loss_args loss_args;



/**
 * DNN BACKEND FUNCTIONS
**/

// DNN initialization function
void DNN_init()
{
  // Layer 0
  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)			l0_in[i] = INPUT[i];
  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)		l0_ker[i] = init_WGT_l0[i];
  // Layer 1
  for(int i=0; i<Tin_C_l1*Tout_C_l1*Tker_H_l1*Tker_W_l1; i++)		l1_ker[i] = init_WGT_l1[i];
  // Layer 2
  //   Pooling kernel (no parameters)
  // Layer 3
  for(int i=0; i<Tin_C_l3*Tout_C_l3*Tker_H_l3*Tker_W_l3; i++)		l3_ker[i] = init_WGT_l3[i];
  // Layer 4
  for(int i=0; i<Tin_C_l4*Tout_C_l4*Tker_H_l4*Tker_W_l4; i++)		l4_ker[i] = init_WGT_l4[i];
  // Layer 5
  for(int i=0; i<Tin_C_l5*Tout_C_l5*Tker_H_l5*Tker_W_l5; i++)		l5_ker[i] = init_WGT_l5[i];
  // Layer 6
  for(int i=0; i<Tin_C_l6*Tout_C_l6*Tker_H_l6*Tker_W_l6; i++)		l6_ker[i] = init_WGT_l6[i];
  // Layer 7
  for(int i=0; i<Tin_C_l7*Tout_C_l7*Tker_H_l7*Tker_W_l7; i++)		l7_ker[i] = init_WGT_l7[i];
  // Layer 8
  for(int i=0; i<Tin_C_l8*Tout_C_l8*Tker_H_l8*Tker_W_l8; i++)		l8_ker[i] = init_WGT_l8[i];
  // Layer 9
  for(int i=0; i<Tin_C_l9*Tout_C_l9*Tker_H_l9*Tker_W_l9; i++)		l9_ker[i] = init_WGT_l9[i];
  // Layer 10
  for(int i=0; i<Tin_C_l10*Tout_C_l10*Tker_H_l10*Tker_W_l10; i++)		l10_ker[i] = init_WGT_l10[i];
  // Layer 11
  for(int i=0; i<Tin_C_l11*Tout_C_l11*Tker_H_l11*Tker_W_l11; i++)		l11_ker[i] = init_WGT_l11[i];
  // Layer 12
  for(int i=0; i<Tin_C_l12*Tout_C_l12*Tker_H_l12*Tker_W_l12; i++)		l12_ker[i] = init_WGT_l12[i];
  // Layer 13
  for(int i=0; i<Tin_C_l13*Tout_C_l13*Tker_H_l13*Tker_W_l13; i++)		l13_ker[i] = init_WGT_l13[i];
  // Layer 14
  for(int i=0; i<Tin_C_l14*Tout_C_l14*Tker_H_l14*Tker_W_l14; i++)		l14_ker[i] = init_WGT_l14[i];
  // Layer 15
  for(int i=0; i<Tin_C_l15*Tout_C_l15*Tker_H_l15*Tker_W_l15; i++)		l15_ker[i] = init_WGT_l15[i];

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


//Connecting ReLU
  // Layer 1
  layer1_in.data = l1_in;
  layer1_in.diff = l1_in_diff;
  layer1_in.dim = Tin_C_l1*Tin_H_l1*Tin_W_l1;
  layer1_in.C = Tin_C_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.W = Tin_W_l1;
  layer1_wgt.data = l1_ker;
  layer1_wgt.diff = l1_ker_diff;
  layer1_wgt.dim = Tin_C_l1*Tout_C_l1*Tker_H_l1*Tker_W_l1;
  layer1_wgt.C = Tin_C_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_out.data = l2_in;
  layer1_out.diff = l2_in_diff;
  layer1_out.dim = Tout_C_l1*Tout_H_l1*Tout_W_l1;
  layer1_out.C = Tout_C_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.W = Tout_W_l1;


//Connecting MaxPool
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


//Connecting conv2d
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


//Connecting ReLU
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


//Connecting conv2d
  // Layer 5
  layer5_in.data = l5_in;
  layer5_in.diff = l5_in_diff;
  layer5_in.dim = Tin_C_l5*Tin_H_l5*Tin_W_l5;
  layer5_in.C = Tin_C_l5;
  layer5_in.H = Tin_H_l5;
  layer5_in.W = Tin_W_l5;
  layer5_wgt.data = l5_ker;
  layer5_wgt.diff = l5_ker_diff;
  layer5_wgt.dim = Tin_C_l5*Tout_C_l5*Tker_H_l5*Tker_W_l5;
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


//Connecting ReLU
  // Layer 8
  layer8_in.data = l8_in;
  layer8_in.diff = l8_in_diff;
  layer8_in.dim = Tin_C_l8*Tin_H_l8*Tin_W_l8;
  layer8_in.C = Tin_C_l8;
  layer8_in.H = Tin_H_l8;
  layer8_in.W = Tin_W_l8;
  layer8_wgt.data = l8_ker;
  layer8_wgt.diff = l8_ker_diff;
  layer8_wgt.dim = Tin_C_l8*Tout_C_l8*Tker_H_l8*Tker_W_l8;
  layer8_wgt.C = Tin_C_l8;
  layer8_wgt.H = Tker_H_l8;
  layer8_wgt.W = Tker_W_l8;
  layer8_out.data = l9_in;
  layer8_out.diff = l9_in_diff;
  layer8_out.dim = Tout_C_l8*Tout_H_l8*Tout_W_l8;
  layer8_out.C = Tout_C_l8;
  layer8_out.H = Tout_H_l8;
  layer8_out.W = Tout_W_l8;


//Connecting conv2d
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


//Connecting ReLU
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


//Connecting conv2d
  // Layer 11
  layer11_in.data = l11_in;
  layer11_in.diff = l11_in_diff;
  layer11_in.dim = Tin_C_l11*Tin_H_l11*Tin_W_l11;
  layer11_in.C = Tin_C_l11;
  layer11_in.H = Tin_H_l11;
  layer11_in.W = Tin_W_l11;
  layer11_wgt.data = l11_ker;
  layer11_wgt.diff = l11_ker_diff;
  layer11_wgt.dim = Tin_C_l11*Tout_C_l11*Tker_H_l11*Tker_W_l11;
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


//Connecting ReLU
  // Layer 14
  layer14_in.data = l14_in;
  layer14_in.diff = l14_in_diff;
  layer14_in.dim = Tin_C_l14*Tin_H_l14*Tin_W_l14;
  layer14_in.C = Tin_C_l14;
  layer14_in.H = Tin_H_l14;
  layer14_in.W = Tin_W_l14;
  layer14_wgt.data = l14_ker;
  layer14_wgt.diff = l14_ker_diff;
  layer14_wgt.dim = Tin_C_l14*Tout_C_l14*Tker_H_l14*Tker_W_l14;
  layer14_wgt.C = Tin_C_l14;
  layer14_wgt.H = Tker_H_l14;
  layer14_wgt.W = Tker_W_l14;
  layer14_out.data = l15_in;
  layer14_out.diff = l15_in_diff;
  layer14_out.dim = Tout_C_l14*Tout_H_l14*Tout_W_l14;
  layer14_out.C = Tout_C_l14;
  layer14_out.H = Tout_H_l14;
  layer14_out.W = Tout_W_l14;


//Connecting linear
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
  layer15_out.data = l15_out;
  layer15_out.diff = l15_out_diff;
  layer15_out.dim = Tout_C_l15*Tout_H_l15*Tout_W_l15;
  layer15_out.C = Tout_C_l15;
  layer15_out.H = Tout_H_l15;
  layer15_out.W = Tout_W_l15;

  // Configure layer structures
  // Layer 0
  l0_args.input = &layer0_in;
  l0_args.coeff = &layer0_wgt;
  l0_args.output = &layer0_out;
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
  l1_args.input = &layer1_in;
  l1_args.output = &layer1_out;
  // Layer 2
  //   Pooling layer (see next section)
  // Layer 3
  l3_args.input = &layer3_in;
  l3_args.coeff = &layer3_wgt;
  l3_args.output = &layer3_out;
  l3_args.skip_in_grad = 0;
  l3_args.Lpad = 1;
  l3_args.Rpad = 1;
  l3_args.Upad = 1;
  l3_args.Dpad = 1;
  l3_args.stride_h = 2;
  l3_args.stride_w = 2;
  l3_args.i2c_buffer = (float*) im2col_buffer;
  l3_args.bt_buffer = (float*) bt_buffer;
  l3_args.HWC = 0;
  l3_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L3;
  l3_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L3;
  l3_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L3;
  l3_args.USE_IM2COL = 1;
  l3_args.USE_DMA_IM2COL = 0;
  // Layer 4
  l4_args.input = &layer4_in;
  l4_args.output = &layer4_out;
  // Layer 5
  l5_args.input = &layer5_in;
  l5_args.coeff = &layer5_wgt;
  l5_args.output = &layer5_out;
  l5_args.skip_in_grad = 0;
  l5_args.Lpad = 1;
  l5_args.Rpad = 1;
  l5_args.Upad = 1;
  l5_args.Dpad = 1;
  l5_args.stride_h = 1;
  l5_args.stride_w = 1;
  l5_args.i2c_buffer = (float*) im2col_buffer;
  l5_args.bt_buffer = (float*) bt_buffer;
  l5_args.HWC = 0;
  l5_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L5;
  l5_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L5;
  l5_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L5;
  l5_args.USE_IM2COL = 1;
  l5_args.USE_DMA_IM2COL = 0;
  // Layer 6
  l6_args.input = &layer6_in;
  l6_args.output = &layer6_out;
  // Layer 7
  l7_args.input = &layer7_in;
  l7_args.coeff = &layer7_wgt;
  l7_args.output = &layer7_out;
  l7_args.skip_in_grad = 0;
  l7_args.Lpad = 1;
  l7_args.Rpad = 1;
  l7_args.Upad = 1;
  l7_args.Dpad = 1;
  l7_args.stride_h = 2;
  l7_args.stride_w = 2;
  l7_args.i2c_buffer = (float*) im2col_buffer;
  l7_args.bt_buffer = (float*) bt_buffer;
  l7_args.HWC = 0;
  l7_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L7;
  l7_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L7;
  l7_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L7;
  l7_args.USE_IM2COL = 1;
  l7_args.USE_DMA_IM2COL = 0;
  // Layer 8
  l8_args.input = &layer8_in;
  l8_args.output = &layer8_out;
  // Layer 9
  l9_args.input = &layer9_in;
  l9_args.coeff = &layer9_wgt;
  l9_args.output = &layer9_out;
  l9_args.skip_in_grad = 0;
  l9_args.Lpad = 1;
  l9_args.Rpad = 1;
  l9_args.Upad = 1;
  l9_args.Dpad = 1;
  l9_args.stride_h = 1;
  l9_args.stride_w = 1;
  l9_args.i2c_buffer = (float*) im2col_buffer;
  l9_args.bt_buffer = (float*) bt_buffer;
  l9_args.HWC = 0;
  l9_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L9;
  l9_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L9;
  l9_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L9;
  l9_args.USE_IM2COL = 1;
  l9_args.USE_DMA_IM2COL = 0;
  // Layer 10
  l10_args.input = &layer10_in;
  l10_args.output = &layer10_out;
  // Layer 11
  l11_args.input = &layer11_in;
  l11_args.coeff = &layer11_wgt;
  l11_args.output = &layer11_out;
  l11_args.skip_in_grad = 0;
  l11_args.Lpad = 1;
  l11_args.Rpad = 1;
  l11_args.Upad = 1;
  l11_args.Dpad = 1;
  l11_args.stride_h = 2;
  l11_args.stride_w = 2;
  l11_args.i2c_buffer = (float*) im2col_buffer;
  l11_args.bt_buffer = (float*) bt_buffer;
  l11_args.HWC = 0;
  l11_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L11;
  l11_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L11;
  l11_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L11;
  l11_args.USE_IM2COL = 1;
  l11_args.USE_DMA_IM2COL = 0;
  // Layer 12
  l12_args.input = &layer12_in;
  l12_args.output = &layer12_out;
  // Layer 13
  l13_args.input = &layer13_in;
  l13_args.coeff = &layer13_wgt;
  l13_args.output = &layer13_out;
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
  l14_args.input = &layer14_in;
  l14_args.output = &layer14_out;
  // Layer 15
  l15_args.input = &layer15_in;
  l15_args.coeff = &layer15_wgt;
  l15_args.output = &layer15_out;
  l15_args.skip_in_grad = 0;
  l15_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L15;
  l15_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L15;
  l15_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L15;

  // Connect blobs to pooling structures
  // Layer 2
  l2_pool_args.input = &layer2_in;
  l2_pool_args.output = &layer2_out;
  l2_pool_args.Hker = Tker_H_l2;
  l2_pool_args.Wker = Tker_W_l2;
  l2_pool_args.Hstride = Tstr_H_l2;
  l2_pool_args.Wstride = Tstr_W_l2;
}


// Forward pass function
void forward()
{
  pulp_conv2d_fp32_fw_cl(&l0_args);
  pulp_relu_fp32_fw_cl(&l1_args);
  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l2_pool_args);
  pulp_conv2d_fp32_fw_cl(&l3_args);
  pulp_relu_fp32_fw_cl(&l4_args);
  pulp_conv2d_fp32_fw_cl(&l5_args);
  pulp_relu_fp32_fw_cl(&l6_args);
  pulp_conv2d_fp32_fw_cl(&l7_args);
  pulp_relu_fp32_fw_cl(&l8_args);
  pulp_conv2d_fp32_fw_cl(&l9_args);
  pulp_relu_fp32_fw_cl(&l10_args);
  pulp_conv2d_fp32_fw_cl(&l11_args);
  pulp_relu_fp32_fw_cl(&l12_args);
  pulp_conv2d_fp32_fw_cl(&l13_args);
  pulp_relu_fp32_fw_cl(&l14_args);
  pulp_linear_fp32_fw_cl(&l15_args);
}

// Backward pass function
void backward()
{
  pulp_linear_fp32_bw_cl(&l15_args);
  pulp_relu_fp32_bw_cl(&l14_args);
  pulp_conv2d_fp32_bw_cl(&l13_args);
  pulp_relu_fp32_bw_cl(&l12_args);
  pulp_conv2d_fp32_bw_cl(&l11_args);
  pulp_relu_fp32_bw_cl(&l10_args);
  pulp_conv2d_fp32_bw_cl(&l9_args);
  pulp_relu_fp32_bw_cl(&l8_args);
  pulp_conv2d_fp32_bw_cl(&l7_args);
  pulp_relu_fp32_bw_cl(&l6_args);
  pulp_conv2d_fp32_bw_cl(&l5_args);
  pulp_relu_fp32_bw_cl(&l4_args);
  pulp_conv2d_fp32_bw_cl(&l3_args);
  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l2_pool_args);
  pulp_relu_fp32_bw_cl(&l1_args);
  pulp_conv2d_fp32_bw_cl(&l0_args);
}

// Compute loss and output gradient
void compute_loss()
{
  loss_args.output = &layer15_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  pulp_MSELoss(&loss_args);
}

// Function to update the network
void update_weights()
{
  struct optim_args opt_l0;
  opt_l0.weights = &layer0_wgt;
  opt_l0.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
  struct optim_args opt_l3;
  opt_l3.weights = &layer3_wgt;
  opt_l3.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l3);
  struct optim_args opt_l5;
  opt_l5.weights = &layer5_wgt;
  opt_l5.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l5);
  struct optim_args opt_l7;
  opt_l7.weights = &layer7_wgt;
  opt_l7.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l7);
  struct optim_args opt_l9;
  opt_l9.weights = &layer9_wgt;
  opt_l9.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l9);
  struct optim_args opt_l11;
  opt_l11.weights = &layer11_wgt;
  opt_l11.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l11);
  struct optim_args opt_l13;
  opt_l13.weights = &layer13_wgt;
  opt_l13.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l13);
  struct optim_args opt_l15;
  opt_l15.weights = &layer15_wgt;
  opt_l15.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l15);
}



/**
 * DATA VISUALIZATION AND CHECK TOOLS
**/

// Function to print FW output
void print_output()
{
  printf("\nLayer 15 output:\n");

  for (int i=0; i<Tout_C_l15*Tout_H_l15*Tout_W_l15; i++)
  {
    printf("%f ", l15_out[i]);
    // Newline when an output row ends
    // if(!(i%Tout_W_l15)) printf("\n");
    // Newline when an output channel ends
    if((i % Tout_W_l15 * Tout_H_l15) == 0) printf("\n");
  }
}

// Function to check post-training output wrt Golden Model (GM)
void check_post_training_output()
{
  int integrity_check = 0;
  integrity_check = verify_tensor(l15_out, REFERENCE_OUTPUT, Tout_C_l15*Tout_H_l15*Tout_W_l15, TOLERANCE);
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
