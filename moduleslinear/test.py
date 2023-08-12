import torch
#from fast_transformers.causal_product import causal_dot_product


q = k = v = torch.randn(5, 10, 10, 10).to(0)
print(causal_dot_product(q, k, v)) # this should produce the right result.