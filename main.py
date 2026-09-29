import torch

def f(q, k, v, mask):
    logits = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5) + mask
    return torch.softmax(logits, dim=-1) @ v

def inspect_backend(graph_module, example_inputs):
    print(graph_module.graph)
    print(graph_module.code)
    return graph_module.forward

captured = torch.compile(f, backend=inspect_backend, fullgraph=True)
q, k, v = torch.randn(8, 16), torch.randn(8, 16), torch.randn(8, 16)
mask = torch.zeros(8, 8)
o = captured(q, k, v, mask)
torch.testing.assert_close(o, f(q, k, v, mask))
