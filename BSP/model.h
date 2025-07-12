#ifndef MICRO_TEMP_NET_H
#define MICRO_TEMP_NET_H

#include <stdint.h>

#define NUM_CLASSES 5  

typedef struct {
    // first layer
    const float* conv1_weight;
    const float* conv1_bias;
    
    // second layer
    const float* conv2_weight;
    const float* conv2_bias;
    
    // first full connrction
    const float* fc1_weight;
    const float* fc1_bias;
    
    // second full connection
    const float* fc2_weight;
    const float* fc2_bias;
} MicroTempNet;

// initialize
void micro_temp_net_init(MicroTempNet* net, 
                        const float* conv1_w, const float* conv1_b,
                        const float* conv2_w, const float* conv2_b,
                        const float* fc1_w, const float* fc1_b,
                        const float* fc2_w, const float* fc2_b);

// airun
void micro_temp_net_predict(const MicroTempNet* net, float* input, float* output);

#endif // MICRO_TEMP_NET_H
						
						
						