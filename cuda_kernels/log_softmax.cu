// https://docs.cupy.dev/en/stable/user_guide/kernel.html

// TODO: Why cant we access float.h?
#ifndef FLT_MAX
#define FLT_MAX 3.4028235e+38f
#endif

extern "C" {
__global__ void log_softmax_forward(float* output, const float* input, int rows, int cols) {
    // Compute the LogSoftmax along the last dimension (cols)
    // Each thread handles one element of the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;

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
        // sum_exp -> softmax_output

        // Compute log softmax
        output[idx] = input[idx] - max_val - logf(sum_exp);
    }
}

__global__ void log_softmax_backward(float* grad_input, const float* grad_output, const float* input, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        // printf("Row %d\n", row);
        int col = idx % cols;
        // printf("Col %d\n", col);

        // Compute max and sum for softmax
        float max_val = -FLT_MAX;
        float sum_exp = 0.0f;
        for (int c = 0; c < cols; ++c) {
            max_val = fmaxf(max_val, input[row * cols + c]);
        }
        // printf("max_val: %f\n", max_val);
        for (int c = 0; c < cols; ++c) {
            sum_exp += expf(input[row * cols + c] - max_val);
        }
        // printf("sum_exp: %f\n", sum_exp);

        // Compute softmax output for this element
        float softmax_output_val = expf(input[idx] - max_val) / sum_exp;


        // printf("softmax_output_val: %f\n", softmax_output_val);

        // Compute sum of gradients for the row
        float grad_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            grad_sum += grad_output[row * cols + c];
        }
        // printf("grad_sum: %f\n", grad_sum);

        // Compute the gradient for each element
        // Grad output is in our testcase 1.0 or 0.0, and because grad output is always 1.0 (in our testcase) it is becoming 0.0
        // grad_input[idx] = softmax_output_val * (grad_output[idx] - grad_sum);
        grad_input[idx] = grad_output[idx] - (softmax_output_val * grad_sum);
        // printf("grad_output[idx]: %f\n", grad_output[idx]);
        // printf("grad_input[idx]: %f\n", grad_input[idx]);
    }
}
}
