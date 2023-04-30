# test_pytorch-install

파이토치를 인스톨하기 위한 각종 시도와 기록들 

## M1 Pro + Macos / Macbook Pro 16 - 20221209

- MBP에서 파이토치 설치 

### Basics

- <https://pytorch.org/>

알아서 잡아준다. M1 프로세서를 확인하기 위해서는 nightly build를 적용하자. 

## Testing Installation

- <https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c>

#### Jupyter 설치 

https://pytorch.org/ 

인스톨 지침을 그대로 따르면 된다. OS 별로 자동으로 잡아 준다. 

```shell
conda install -c conda-forge jupyter jupyterlab
```

- `$ jupyter lab`으로 주피터 실행 
- VS CODE를 쓰자.

### 설치 확인 

```python
import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
```

- `true`, `true`로 출력되면 잘 안착된 것 

### 실행 확인

```python
dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

## x86 + CUDA + Windows 11 / Intel i9 - 20230430

### nvidia 장비 확인 

- CUDA 버전도 확인 가능하다. 

```shell
> nvidia-smi
```

### CUDA 확인 

```python
>>> import torch
>>> import os
>>> os.environ["CUDA_VISIBLE_DEVICES"] = "1"

>>> print(f"CUDA availability: {torch.cuda.is_available()}")
>>> print(f"# of CUDA: {torch.cuda.device_count()}")

>>> print(f"Currently selected CUDA devie: {torch.cuda.current_device()}")
>>> print(torch.cuda.device(0))

>>> print(f"Name of GPU: {torch.cuda.get_device_name(0)}")

```

```raw
CUDA availability: True
# of CUDA: 1
Currently selected CUDA devie: 0
<torch.cuda.device object at 0x000002174AED05D0>
Name of GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```
