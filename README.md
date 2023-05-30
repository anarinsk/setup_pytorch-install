# test_pytorch-install

파이토치를 인스톨하기 위한 각종 시도와 기록들 

## [`20230530`] Rye + VS Code

[Rye](https://rye-up.com/)를 먼저 설치하자. 

github에서 pull한 `test_pytorch-install` 폴더로 들어가자. 

### CPU, MPS 활용할 경우 

```shell
> rye init 
> rye add ipykernel torch torchvision torchaudio
> rye sync 
```

### CUDA 

쿠다를 활용할 경우는 별도의 패키지 의존성이 필요하다. 이 녀석을 어떻게 잡아주면 될까? 

`pyproject.toml`을 열고, 아래 내용을 추가하도록 하자. name은 임의로 정해도 된다. 

```toml
[[tool.rye.sources]]
name = "pytorch-cuda118"
url = "https://download.pytorch.org/whl/cu118"
type = "index"
```

새롭게 `.lock`을 생성하고, 패키지를 설치하자. 

```shell
> rye add torch torchvision torchaudio

Added torch==2.0.1+cu118 as regular dependency
Added torchvision==0.15.2+cu118 as regular dependency
Added torchaudio==2.0.2+cu118 as regular dependency

> rye sync
```

`.lock`을 생성하는 데 조금 시간이 걸릴 수 있다. 기다리면 `.lock`이 생성되고 패키지 설치가 시작된다. 

VS Code의 가상 환경을 `.venv`로 잡아주자. 이를 통해 해당 폴더의 `.venv`에 깔린 파이썬 가상 환경을 커널로 부리게 된다. 

`test_working-example.ipynb`를 실행해서 원하는 버전의 pytorch가 설치되었는지 확인하도록 하자. cpu, cuda, mps(macos)를 각각 확인할 수 있어야 한다. 

## [`20221209`] M1 Pro + Macos / Macbook Pro 16

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
