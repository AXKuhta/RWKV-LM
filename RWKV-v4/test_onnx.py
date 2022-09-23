from transformers import PreTrainedTokenizerFast
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from torch.nn import functional as F
import numpy as np
import torch
import array
import time

def lprint(txt):
	print(txt, end='', flush=True)

def sample_logits(out, temperature=1.0, top_p=0.7):
	probs = F.softmax(torch.tensor(out), dim=-1)
	sorted_probs, _ = torch.sort(probs, descending=True)

	cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
	cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
	probs[probs < cutoff] = 0

	if temperature != 1.0:
		probs = probs.pow(1.0 / temperature)

	return torch.multinomial(probs, num_samples=1)[0]

def read_emb(token):
	embds.seek(4*n_embd*token)
	floats = array.array("f", embds.read(4*n_embd))

	return floats.tolist()

def onnx_rnn_run(ctx):
	xx_att = torch.zeros(12, 768).tolist()
	aa_att = torch.zeros(12, 768).tolist()
	bb_att = torch.zeros(12, 768).tolist()
	pp_att = (torch.zeros(12, 768) - 1e30).tolist()
	xx_ffn = torch.zeros(12, 768).tolist()

	ptx = [ ctx.pop(0) ]

	start = time.time_ns()

	for i in range(64):
		emb = read_emb(ptx[-1])

		inputs = { "emb": emb, "xx_att": xx_att, "aa_att": aa_att, "bb_att": bb_att, "pp_att": pp_att, "xx_ffn": xx_ffn }

		outputs = session.run(output_names=["x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"], input_feed=inputs)
		state = outputs[0] # [50277]

		# [12][768]
		xx_att = outputs[1]
		aa_att = outputs[2]
		bb_att = outputs[3]
		pp_att = outputs[4]
		xx_ffn = outputs[5]

		char = sample_logits(state)
		char = char.item()

		if len(ctx) > 0:
			# Outputs produced during the hidden state init sequence may be interesting to observe
			# lprint( tokenizer.decode(ptx) + " ===>" + tokenizer.decode(char) )
			ptx.append( ctx.pop(0) )
		else:
			lprint( tokenizer.decode(char) )
			ptx.append(char)

	stop = time.time_ns()

	print("\n", (stop - start)/1000/1000/64, "ms per token")

opt = SessionOptions()
#opt.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
session = InferenceSession("rwkv.onnx", opt)
embds = open("emb.weight.bin", "rb")

text = """\nIn a shocking finding,"""
ctx = tokenizer.encode(text)

n_layer, n_embd = session.get_inputs()[1].shape
ctx_len = session.get_inputs()[0].shape[0]
print(" n_layer:", n_layer)
print(" n_embd:", n_embd)
print(" ctx_len:", ctx_len)

print("Tokens in context:", len(ctx))
lprint( tokenizer.decode(ctx) )
onnx_rnn_run(ctx)

