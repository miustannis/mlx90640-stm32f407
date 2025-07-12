
#include "model.h"
#include <string.h>
#include <math.h>

// ReLU
static void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

// 2d convolution stride = 2
static void conv2d_stride2(const float* input, const float* weights, const float* bias,
                          float* output, int in_channels, int out_channels,
                          int height, int width, int kernel_size) {
    int out_h = height / 2;
    int out_w = width / 2;
    
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias[oc];
                int ih = oh * 2;
                int iw = ow * 2;
                
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int input_idx = ic * height * width + (ih + kh) * width + (iw + kw);
                            int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                           ic * kernel_size * kernel_size + 
                                           kh * kernel_size + kw;
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
                int output_idx = oc * out_h * out_w + oh * out_w + ow;
                output[output_idx] = sum;
            }
        }
    }
}

// fc
static void dense(const float* input, const float* weights, const float* bias,
                float* output, int in_features, int out_features) {
    for (int i = 0; i < out_features; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_features; j++) {
            output[i] += input[j] * weights[i * in_features + j];
        }
    }
}

// initialize
void micro_temp_net_init(MicroTempNet* net, 
                        const float* conv1_w, const float* conv1_b,
                        const float* conv2_w, const float* conv2_b,
                        const float* fc1_w, const float* fc1_b,
                        const float* fc2_w, const float* fc2_b) {
    net->conv1_weight = conv1_w;
    net->conv1_bias = conv1_b;
    net->conv2_weight = conv2_w;
    net->conv2_bias = conv2_b;
    net->fc1_weight = fc1_w;
    net->fc1_bias = fc1_b;
    net->fc2_weight = fc2_w;
    net->fc2_bias = fc2_b;
}

// predict
void micro_temp_net_predict(const MicroTempNet* net, float* input, float* output) {
    //  [1,32,24] -> [4,16,12]
    float conv1_out[4 * 16 * 12];
    conv2d_stride2(input, net->conv1_weight, net->conv1_bias, conv1_out, 1, 4, 32, 24, 3);
    relu(conv1_out, 4 * 16 * 12);
    
    //  [4,16,12] -> [8,8,6]
    float conv2_out[8 * 8 * 6];
    conv2d_stride2(conv1_out, net->conv2_weight, net->conv2_bias, conv2_out, 4, 8, 16, 12, 3);
    relu(conv2_out, 8 * 8 * 6);
    
    // [8,8,6] -> 384
    float flattened[8 * 8 * 6];
    memcpy(flattened, conv2_out, sizeof(flattened));
    
    // 384->8
    float fc1_out[8];
    dense(flattened, net->fc1_weight, net->fc1_bias, fc1_out, 8 * 8 * 6, 8);
    relu(fc1_out, 8);
    
    // 8->5
    dense(fc1_out, net->fc2_weight, net->fc2_bias, output, 8, NUM_CLASSES);
}
