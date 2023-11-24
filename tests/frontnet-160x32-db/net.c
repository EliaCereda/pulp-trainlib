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

// Define DNN layer structures
PI_L1 struct vect_sum_args vect_sum_args;
PI_L2 struct Conv2D_args l0_args;
PI_L2 struct act_args l1_args;
PI_L2 struct Conv2D_args l3_args;
PI_L2 struct act_args l4_args;
PI_L2 struct Conv2D_args l5_args;
PI_L2 struct act_args l6_args;
PI_L2 struct Conv2D_args l7_args;
PI_L2 struct act_args l8_args;
PI_L2 struct Conv2D_args l9_args;
PI_L2 struct act_args l10_args;
PI_L2 struct Conv2D_args l11_args;
PI_L2 struct act_args l12_args;
PI_L2 struct Conv2D_args l13_args;
PI_L2 struct act_args l14_args;
PI_L2 struct Linear_args l15_args;

// Define Pooling Structures
PI_L2 struct pool_args l2_pool_args;

// Define kernel tensors
PI_L2 float l0_ker[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L2 float l2_ker[1];
PI_L2 float l3_ker[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L2 float l4_ker[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L2 float l6_ker[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L2 float l7_ker[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_ker[Tin_C_l8 * Tout_C_l8 * Tker_H_l8 * Tker_W_l8];
PI_L2 float l9_ker[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker[Tin_C_l11 * Tout_C_l11 * Tker_H_l11 * Tker_W_l11];
PI_L2 float l12_ker[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L2 float l15_ker[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];

// Define kernel grad tensors
PI_L2 float l0_ker_diff[Tin_C_l0 * Tout_C_l0 * Tker_H_l0 * Tker_W_l0];
PI_L2 float l1_ker_diff[Tin_C_l1 * Tout_C_l1 * Tker_H_l1 * Tker_W_l1];
PI_L2 float l2_ker_diff[1];
PI_L2 float l3_ker_diff[Tin_C_l3 * Tout_C_l3 * Tker_H_l3 * Tker_W_l3];
PI_L2 float l4_ker_diff[Tin_C_l4 * Tout_C_l4 * Tker_H_l4 * Tker_W_l4];
PI_L2 float l5_ker_diff[Tin_C_l5 * Tout_C_l5 * Tker_H_l5 * Tker_W_l5];
PI_L2 float l6_ker_diff[Tin_C_l6 * Tout_C_l6 * Tker_H_l6 * Tker_W_l6];
PI_L2 float l7_ker_diff[Tin_C_l7 * Tout_C_l7 * Tker_H_l7 * Tker_W_l7];
PI_L2 float l8_ker_diff[Tin_C_l8 * Tout_C_l8 * Tker_H_l8 * Tker_W_l8];
PI_L2 float l9_ker_diff[Tin_C_l9 * Tout_C_l9 * Tker_H_l9 * Tker_W_l9];
PI_L2 float l10_ker_diff[Tin_C_l10 * Tout_C_l10 * Tker_H_l10 * Tker_W_l10];
PI_L2 float l11_ker_diff[Tin_C_l11 * Tout_C_l11 * Tker_H_l11 * Tker_W_l11];
PI_L2 float l12_ker_diff[Tin_C_l12 * Tout_C_l12 * Tker_H_l12 * Tker_W_l12];
PI_L2 float l13_ker_diff[Tin_C_l13 * Tout_C_l13 * Tker_H_l13 * Tker_W_l13];
PI_L2 float l14_ker_diff[Tin_C_l14 * Tout_C_l14 * Tker_H_l14 * Tker_W_l14];
PI_L2 float l15_ker_diff[Tin_C_l15 * Tout_C_l15 * Tker_H_l15 * Tker_W_l15];

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
PI_L2 float l15_out[Tout_C_l15 * Tout_H_l15 * Tout_W_l15];

// Define IM2COL buffer for all the convolutions
PI_L1 float im2col_buffer[Tout_C_l0*Tker_H_l0*Tker_W_l0*Tin_H_l0*Tin_W_l0];

// Define transposition / block transposition buffer for all conv2d and PW layers
PI_L1 float bt_buffer[Tin_C_l13*Tout_C_l13*Tker_H_l13*Tker_W_l13];

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
PI_L2 float l15_out_diff[Tout_C_l15 * Tout_H_l15 * Tout_W_l15];

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
  l1_args.output = &out;
  // Layer 2
  //   Pooling layer (see next section)
  // Layer 3
  l3_args.input = &in;
  l3_args.coeff = &wgt;
  l3_args.output = &out;
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
  l4_args.input = &in;
  l4_args.output = &out;
  // Layer 5
  l5_args.input = &in;
  l5_args.coeff = &wgt;
  l5_args.output = &out;
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
  l8_args.input = &in;
  l8_args.output = &out;
  // Layer 9
  l9_args.input = &in;
  l9_args.coeff = &wgt;
  l9_args.output = &out;
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
  l10_args.input = &in;
  l10_args.output = &out;
  // Layer 11
  l11_args.input = &in;
  l11_args.coeff = &wgt;
  l11_args.output = &out;
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
  l14_args.output = &out;
  // Layer 15
  l15_args.input = &in;
  l15_args.coeff = &wgt;
  l15_args.output = &out;
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
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 1 (ReLU, 1, 0, 1)
	get_dim(&layer1_out, &d0_blob);
	get_dim(&layer2_wgt, &w0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer0_out.data, 4*layer0_out.dim);
	load((uint32_t) layer2_wgt.data, (uint32_t) w0_blob.data, 4*layer2_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 2 (MaxPool, 0, 1, 0)
	copy_struct_param((uint32_t) &l2_args, (uint32_t) &MaxPool_args, sizeof(MaxPool_args));
	get_dim(&layer2_out, &d1_blob);
	get_dim(&layer3_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer1_out.data, 4*layer1_out.dim);
	load((uint32_t) layer3_wgt.data, (uint32_t) w1_blob.data, 4*layer3_wgt.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l2_pool_args);

	// Layer 3 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l3_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer3_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer2_out.data, 4*layer2_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 4 (ReLU, 0, 1, 0)
	get_dim(&layer4_out, &d1_blob);
	get_dim(&layer5_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer3_out.data, 4*layer3_out.dim);
	load((uint32_t) layer5_wgt.data, (uint32_t) w1_blob.data, 4*layer5_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 5 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l5_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer5_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer4_out.data, 4*layer4_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

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
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer6_out.data, 4*layer6_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 8 (ReLU, 0, 1, 0)
	get_dim(&layer8_out, &d1_blob);
	get_dim(&layer9_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer7_out.data, 4*layer7_out.dim);
	load((uint32_t) layer9_wgt.data, (uint32_t) w1_blob.data, 4*layer9_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 9 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l9_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer9_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer8_out.data, 4*layer8_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 10 (ReLU, 0, 1, 0)
	get_dim(&layer10_out, &d1_blob);
	get_dim(&layer11_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer9_out.data, 4*layer9_out.dim);
	load((uint32_t) layer11_wgt.data, (uint32_t) w1_blob.data, 4*layer11_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 11 (conv2d, 1, 0, 1)
	copy_struct_param((uint32_t) &l11_args, (uint32_t) &conv2d_args, sizeof(conv2d_args));
	get_dim(&layer11_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer10_out.data, 4*layer10_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

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
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer12_out.data, 4*layer12_out.dim);
	pulp_conv2d_fp32_fw_cl(&conv2d_args);

	// Layer 14 (ReLU, 0, 1, 0)
	get_dim(&layer14_out, &d1_blob);
	get_dim(&layer15_wgt, &w1_blob);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	pi_cl_dma_flush();
	store((uint32_t) d0_blob.data, (uint32_t) layer13_out.data, 4*layer13_out.dim);
	load((uint32_t) layer15_wgt.data, (uint32_t) w1_blob.data, 4*layer15_wgt.dim);
	pulp_relu_fp32_fw_cl(&act_args);

	// Layer 15 (linear, 1, 0, 1)
	copy_struct_param((uint32_t) &l15_args, (uint32_t) &linear_args, sizeof(linear_args));
	get_dim(&layer15_out, &d0_blob);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	pi_cl_dma_flush();
	store((uint32_t) d1_blob.data, (uint32_t) layer14_out.data, 4*layer14_out.dim);
	pulp_linear_fp32_fw_cl(&linear_args);

	store((uint32_t) out.data, (uint32_t) layer15_out.data, 4*layer15_out.dim);
	pi_cl_dma_cmd_wait(cmd_store);
}

// Backward pass function
void backward()
{

	// Layer 15 (linear, 0, 1, 0)
	copy_struct_param((unsigned int) &l15_args, (unsigned int) &linear_args, sizeof(l15_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer15_out.diff, 4*layer15_out.dim);
	pulp_linear_fp32_bw_param_grads_cl(&linear_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer15_wgt.diff, 4*layer15_wgt.dim);
	pulp_linear_fp32_bw_input_grads_cl(&linear_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer14_in, &d0_blob);
	load((uint32_t) layer14_in.data, (uint32_t) d0_blob.data, 4*layer14_in.dim);
	pi_cl_dma_flush();

	// Layer 14 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer14_out.diff, 4*layer14_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
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

	// Layer 11 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l11_args, (unsigned int) &conv2d_args, sizeof(l11_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer11_out.diff, 4*layer11_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer11_wgt.diff, 4*layer11_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer10_in, &d0_blob);
	load((uint32_t) layer10_in.data, (uint32_t) d0_blob.data, 4*layer10_in.dim);
	pi_cl_dma_flush();

	// Layer 10 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer10_out.diff, 4*layer10_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer9_in, &d1_blob);
	load((uint32_t) layer9_in.data, (uint32_t) d1_blob.data, 4*layer9_in.dim);
	get_dim( &layer9_wgt, &w1_blob);
	load((uint32_t) layer9_wgt.data, (uint32_t) w1_blob.data, 4*layer9_wgt.dim);
	pi_cl_dma_flush();

	// Layer 9 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l9_args, (unsigned int) &conv2d_args, sizeof(l9_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer9_out.diff, 4*layer9_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer9_wgt.diff, 4*layer9_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer8_in, &d0_blob);
	load((uint32_t) layer8_in.data, (uint32_t) d0_blob.data, 4*layer8_in.dim);
	pi_cl_dma_flush();

	// Layer 8 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer8_out.diff, 4*layer8_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
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

	// Layer 5 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l5_args, (unsigned int) &conv2d_args, sizeof(l5_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer5_out.diff, 4*layer5_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer5_wgt.diff, 4*layer5_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer4_in, &d0_blob);
	load((uint32_t) layer4_in.data, (uint32_t) d0_blob.data, 4*layer4_in.dim);
	pi_cl_dma_flush();

	// Layer 4 (ReLU, 1, 0, 1)
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer4_out.diff, 4*layer4_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer3_in, &d1_blob);
	load((uint32_t) layer3_in.data, (uint32_t) d1_blob.data, 4*layer3_in.dim);
	get_dim( &layer3_wgt, &w1_blob);
	load((uint32_t) layer3_wgt.data, (uint32_t) w1_blob.data, 4*layer3_wgt.dim);
	pi_cl_dma_flush();

	// Layer 3 (conv2d, 0, 1, 0)
	copy_struct_param((unsigned int) &l3_args, (unsigned int) &conv2d_args, sizeof(l3_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer3_out.diff, 4*layer3_out.dim);
	pulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);
	store((uint32_t) w1_blob.diff, (uint32_t) layer3_wgt.diff, 4*layer3_wgt.dim);
	pulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer2_in, &d0_blob);
	load((uint32_t) layer2_in.data, (uint32_t) d0_blob.data, 4*layer2_in.dim);
	get_dim( &layer2_wgt, &w0_blob);
	load((uint32_t) layer2_wgt.data, (uint32_t) w0_blob.data, 4*layer2_wgt.dim);
	pi_cl_dma_flush();

	// Layer 2 (MaxPool, 1, 0, 1)
	copy_struct_param((unsigned int) &l2_args, (unsigned int) &MaxPool_args, sizeof(l2_args));
	pi_cl_dma_cmd_wait(cmd_struct);
	in = d0_blob;
	wgt = w0_blob;
	out = d1_blob;
	store((uint32_t) d1_blob.diff, (uint32_t) layer2_out.diff, 4*layer2_out.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l2_pool_args);
	store((uint32_t) w0_blob.diff, (uint32_t) layer2_wgt.diff, 4*layer2_wgt.dim);
	pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l2_pool_args);
	pi_cl_dma_cmd_wait(cmd_store);
	get_dim( &layer1_in, &d1_blob);
	load((uint32_t) layer1_in.data, (uint32_t) d1_blob.data, 4*layer1_in.dim);
	pi_cl_dma_flush();

	// Layer 1 (ReLU, 0, 1, 0)
	in = d1_blob;
	wgt = w1_blob;
	out = d0_blob;
	store((uint32_t) d0_blob.diff, (uint32_t) layer1_out.diff, 4*layer1_out.dim);
	pulp_relu_fp32_bw_cl(&act_args);
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
	store((uint32_t) out.diff, (uint32_t) layer15_out.diff, 4*OUT_SIZE);
}

// Function to update the network
void update_weights()
{
	// Creates all structures needed
	struct optim_args opt_l0;
	struct optim_args opt_l3;
	struct optim_args opt_l5;
	struct optim_args opt_l7;
	struct optim_args opt_l9;
	struct optim_args opt_l11;
	struct optim_args opt_l13;
	struct optim_args opt_l15;

	pi_cl_dma_flush();
	get_dim(&layer0_wgt, &d0_blob);
	opt_l0.weights = &d0_blob;
	opt_l0.learning_rate = LEARNING_RATE;
	load((uint32_t) layer0_wgt.data, (uint32_t) d0_blob.data, 4*layer0_wgt.dim);
	load((uint32_t) layer0_wgt.diff, (uint32_t) d0_blob.diff, 4*layer0_wgt.dim);

	pi_cl_dma_flush();
	get_dim(&layer3_wgt, &d1_blob);
	opt_l3.weights = &d1_blob;
	opt_l3.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer3_wgt.data, (uint32_t) d1_blob.data, 4*layer3_wgt.dim);
	load((uint32_t) layer3_wgt.diff, (uint32_t) d1_blob.diff, 4*layer3_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
	store((uint32_t) d0_blob.data, (uint32_t) layer0_wgt.data, 4*layer0_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer5_wgt, &d0_blob);
	opt_l5.weights = &d0_blob;
	opt_l5.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer5_wgt.data, (uint32_t) d0_blob.data, 4*layer5_wgt.dim);
	load((uint32_t) layer5_wgt.diff, (uint32_t) d0_blob.diff, 4*layer5_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l3);
	store((uint32_t) d1_blob.data, (uint32_t) layer3_wgt.data, 4*layer3_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer7_wgt, &d1_blob);
	opt_l7.weights = &d1_blob;
	opt_l7.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer7_wgt.data, (uint32_t) d1_blob.data, 4*layer7_wgt.dim);
	load((uint32_t) layer7_wgt.diff, (uint32_t) d1_blob.diff, 4*layer7_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l5);
	store((uint32_t) d0_blob.data, (uint32_t) layer5_wgt.data, 4*layer5_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer9_wgt, &d0_blob);
	opt_l9.weights = &d0_blob;
	opt_l9.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer9_wgt.data, (uint32_t) d0_blob.data, 4*layer9_wgt.dim);
	load((uint32_t) layer9_wgt.diff, (uint32_t) d0_blob.diff, 4*layer9_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l7);
	store((uint32_t) d1_blob.data, (uint32_t) layer7_wgt.data, 4*layer7_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
	get_dim(&layer11_wgt, &d1_blob);
	opt_l11.weights = &d1_blob;
	opt_l11.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer11_wgt.data, (uint32_t) d1_blob.data, 4*layer11_wgt.dim);
	load((uint32_t) layer11_wgt.diff, (uint32_t) d1_blob.diff, 4*layer11_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l9);
	store((uint32_t) d0_blob.data, (uint32_t) layer9_wgt.data, 4*layer9_wgt.dim);
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
	get_dim(&layer15_wgt, &d1_blob);
	opt_l15.weights = &d1_blob;
	opt_l15.learning_rate = LEARNING_RATE;
	pi_cl_dma_cmd_wait(cmd_store);
	load((uint32_t) layer15_wgt.data, (uint32_t) d1_blob.data, 4*layer15_wgt.dim);
	load((uint32_t) layer15_wgt.diff, (uint32_t) d1_blob.diff, 4*layer15_wgt.dim);
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l13);
	store((uint32_t) d0_blob.data, (uint32_t) layer13_wgt.data, 4*layer13_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);

	pi_cl_dma_flush();
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l15);
	store((uint32_t) d1_blob.data, (uint32_t) layer15_wgt.data, 4*layer15_wgt.dim);
	pi_cl_dma_cmd_wait(cmd_store);
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
    if(!(i%Tout_W_l15*Tout_H_l15)) printf("\n");
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
