// https://docs.cupy.dev/en/stable/user_guide/kernel.html

// TODO: Why cant we access float.h?
#ifndef FLT_MAX
#define FLT_MAX 3.4028235e+38f
#endif

extern "C" {
__global__ void softmax_forward(float* output, const float* input, int rows, int cols) {


    // Compute the Softmax along the last dimension (cols)
    // Each thread handles one element of the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        // int col = idx % cols;

        // Compute max for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            max_val = fmaxf(max_val, input[row * cols + c]);
        }

        // Compute sum of exp(shifted_logits)
        float sum_exp = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum_exp += expf(input[row * cols + c] - max_val);
        }

        // Compute softmax
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}


__global__ void softmax_backward(float* grad_input, const float* grad_output, const float* input, int rows, int cols) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;

    if (row < rows) {

        float dot_product_sum = 0.0f;

        for (int c = 0; c < cols; c++) {
            int index = row * cols + c;
            dot_product_sum += grad_output[index] * input[index];
        }


        for (int c = 0; c < cols; c++) {
            int index = row * cols + c;
            grad_input[index] = input[index] * (grad_output[index] - dot_product_sum);
        }
    }
}
}
