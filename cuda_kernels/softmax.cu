// https://docs.cupy.dev/en/stable/user_guide/kernel.html

// TODO: Why cant we access float.h?
#ifndef FLT_MAX
#define FLT_MAX 3.4028235e+38f
#endif

extern "C" {
__global__ void softmax_forward(float* output, const float* input, int rows, int cols) {
    extern __shared__ float shared_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    
    // Find max and compute exp sum
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = input[row * cols + c];
        thread_max = fmaxf(thread_max, val);
    }
    
    // Reduce max within block
    shared_data[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared_data[0];
    __syncthreads();  // Ensure all threads have the max value

    // Compute exp sum
    for (int c = tid; c < cols; c += blockDim.x) {
        float exp_val = expf(input[row * cols + c] - max_val);
        thread_sum += exp_val;
        output[row * cols + c] = exp_val;  // Store intermediate result
    }

    // Reduce sum within block
    shared_data[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float sum_exp = shared_data[0];

    // Normalize with sum
    for (int c = tid; c < cols; c += blockDim.x) {
        output[row * cols + c] /= sum_exp;
    }
}


__global__ void softmax_backward(float* grad_input, const float* grad_output, const float* softmax_output, int rows, int cols) {
    extern __shared__ float shared_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    
    // Compute dot product of grad_output and softmax_output
    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = row * cols + c;
        thread_sum += grad_output[idx] * softmax_output[idx];
    }
    
    // Reduce sum within block
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float sum = shared_data[0];
    
    // Compute gradient
    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = row * cols + c;
        grad_input[idx] = softmax_output[idx] * (grad_output[idx] - sum);
    }
}

}
