# %% 
import torch

# %% Tensor zero wymiarowy
x = torch.tensor(3.0)
print(x, x.shape)

# %% 
x = torch.tensor([[1, 2], [3, 4]])
print(x, x.shape)
print(x.grad)

# %% 
print(x + 2)  # dodawanie
# %% 
print(x * 2)  # mnożenie   
# %% 
print(x.T)    # transponowanie

# %% 
x = torch.tensor(3.0, requires_grad=True)
print(x, x.shape)

# %% 
print(x.grad)

# %% 
y = x + 2
print(y.data)

# %% 
z = y ** 2
print(z.data)

# %% 
z.backward()
print(x.grad)


