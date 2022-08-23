import torch

class MATMUL(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, mat, vec):
		return mat @ vec

def export():
	model = MATMUL()

	mat = torch.rand([768, 768])
	vec = torch.rand([768, 1])

	torch.onnx.export(model, args=(mat, vec), f="matmul.onnx", input_names = ["mat", "vec"], output_names = ["x"], verbose=True)


