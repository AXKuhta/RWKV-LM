from transformers import PreTrainedTokenizerFast
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from torch.nn import functional as F
import numpy as np
import torch
import array
import json
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
	xx_att = []
	aa_att = []
	bb_att = []
	pp_att = []
	xx_ffn = []

	for i in range(n_layer):
		xx_att.append( torch.zeros(n_embd).tolist() )
		aa_att.append( torch.zeros(n_embd).tolist() )
		bb_att.append( torch.zeros(n_embd).tolist() )
		pp_att.append( (torch.zeros(n_embd) - 1e30).tolist() )
		xx_ffn.append( torch.zeros(n_embd).tolist() )

	ptx = [ ctx.pop(0) ]

	start = time.time_ns()

	for i in range(64):
		emb = read_emb(ptx[-1])

		for i in range(n_layer):
			inputs = { "emb": emb, "xx_att": xx_att[i], "aa_att": aa_att[i], "bb_att": bb_att[i], "pp_att": pp_att[i], "xx_ffn": xx_ffn[i] }
			outputs = layers[i].run(output_names=["x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"], input_feed=inputs)

			emb = outputs[0]
			xx_att[i] = outputs[1]
			aa_att[i] = outputs[2]
			bb_att[i] = outputs[3]
			pp_att[i] = outputs[4]
			xx_ffn[i] = outputs[5]

		state = outputs[0] # [50277]
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


params_f = open("rwkv.json", "r")
params = json.load(params_f)
params_f.close()

n_layer, n_embd, ctx_len = (params["n_layer"], params["n_embd"], params["ctx_len"])

print(" n_layer:", n_layer)
print(" n_embd:", n_embd)
print(" ctx_len:", ctx_len)

tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
embds = open("emb.weight.bin", "rb")

layers = []

opt = SessionOptions()
#opt.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL

for i in range(n_layer):
	layers.append( InferenceSession(f"rwkv.{i}.onnx", opt) )

text = """\nIn a shocking finding,"""
ctx = tokenizer.encode(text)

print("Tokens in context:", len(ctx))
lprint( tokenizer.decode(ctx) )
onnx_rnn_run(ctx)

