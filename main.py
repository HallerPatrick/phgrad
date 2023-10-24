from phgrad.engine import MLP, Scalar, softmax_scalars

import torch

def main():

    a = Scalar(-4.0)
    b = Scalar(2.0)

    c = softmax_scalars([a, b])
    c[0].grad = 1.0
    c[1].grad = 1.0
    c[0].backward()
    c[1].backward()


    print("phgrad Softmax values:")
    print(a.value, b.value)

    
    print("phgrad Gradients:")
    print(a.grad, b.grad)

    a2 = torch.tensor([-4.0, 2.0], dtype=torch.float64, requires_grad=True)
    c2 = torch.softmax(a2, dim=0)

    print("PyTorch Softmax values:")
    print(c2[0].item(), c2[1].item())

    grad_out = torch.tensor([1.0, 0.0], dtype=torch.float64)
    c2.backward(grad_out, retain_graph=True)
    print("Gradient with respect to first element:")
    print(a2.grad)

    # Reset gradients
    a2.grad.zero_()
    
    grad_out = torch.tensor([0.0, 1.0], dtype=torch.float64)
    c2.backward(grad_out, retain_graph=True)
    print("Gradient with respect to second element:")
    print(a2.grad)



if __name__ == "__main__":
    main()
