// PULP Defines
#define STACK_SIZE      4096

// Tolerance to check updated output
#define TOLERANCE 1e-12

// Training functions
void DNN_init();
void compute_loss();
void update_weights();
void forward();
void backward();
void net_step();

// Print and check functions
void print_output();
void check_post_training_output();

// DMA managment functions
void load_input(void * src_blob, uint8_t data_diff_both);
void load_output(void * src_blob, uint8_t data_diff_both);
void load_coeff(void * src_blob, uint8_t data_diff_both);
void swap_in_out();
void store_output(void * dest_blob, uint8_t data_diff_both);
void store_input(void * dest_blob, uint8_t data_diff_both);
void store_coeff(void * dest_blob, uint8_t data_diff_both);
void copy_struct_param(unsigned int from, unsigned int to, int size);
void get_input_dim(void * b);
void get_output_dim(void * b);
void get_weight_dim(void * b);
void reset_arguments();
void update_blob();
void reset_dim();
void dma_handler(uint8_t do_store, uint8_t do_load, void * src_store, void * dst_store, void * src_load, void * dst_load);
void load(uint32_t src, uint32_t dst, int dim);
void store(uint32_t src, uint32_t dst, int dim);
void update();
void get_dim(void * src_blob, void * dst_blob);
#define MAX_SIZE 597504
