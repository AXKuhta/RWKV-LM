########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import torch
import os
from torch.nn import functional as F

from transformers import PreTrainedTokenizerFast

MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
n_layer = 12
n_embd = 768
ctx_len = 1024

# MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
# n_layer = 24
# n_embd = 1024
# ctx_len = 1024

os.environ['RWKV_FLOAT_MODE'] = 'fp32'
os.environ['RWKV_RUN_DEVICE'] = 'cpu'
model_type = 'RWKV'

from src.model_run import RWKV_RNN

np.set_printoptions(precision=4, suppress=True, linewidth=200)

tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
context = '\nIn a shocking finding,'

##############################################################################################################

def sample_logits(out, temperature=1.0, top_p=0.7):
    probs = F.softmax(torch.tensor(out), dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    return torch.multinomial(probs, num_samples=1)[0]

def rnn_test(context):
	model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)

	xx_att = torch.zeros(n_layer, n_embd)
	aa_att = torch.zeros(n_layer, n_embd)
	bb_att = torch.zeros(n_layer, n_embd)
	pp_att = torch.zeros(n_layer, n_embd) - 1e30
	xx_ffn = torch.zeros(n_layer, n_embd)

	ctx = tokenizer.encode(context)
	emb = 0

	print(context, end='', flush=True)

	for i in range(64):
		if len(ctx) > 0:
			emb = model.w.emb.weight[ ctx.pop(0) ]

		x, xx_att, aa_att, bb_att, pp_att, xx_ffn  = model( emb, xx_att, aa_att, bb_att, pp_att, xx_ffn )

		if len(ctx) == 0:
			char = sample_logits( x.tolist() )
			char = char.item()
			emb = model.w.emb.weight[char]

			print(tokenizer.decode(char), end='', flush=True)

	print("")

def jit_rnn_test(context):
	model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)

	xx_att = torch.zeros(n_layer, n_embd)
	aa_att = torch.zeros(n_layer, n_embd)
	bb_att = torch.zeros(n_layer, n_embd)
	pp_att = torch.zeros(n_layer, n_embd) - 1e30
	xx_ffn = torch.zeros(n_layer, n_embd)

	emb = torch.rand(n_embd)

	jit = torch.jit.trace(model.eval(), (emb, xx_att, aa_att, bb_att, pp_att, xx_ffn))
	jit = torch.jit.optimize_for_inference(jit)

	ctx = tokenizer.encode(context)
	emb = 0

	print(context, end='', flush=True)

	for i in range(64):
		if len(ctx) > 0:
			emb = model.w.emb.weight[ ctx.pop(0) ]

		x, xx_att, aa_att, bb_att, pp_att, xx_ffn  = jit( emb, xx_att, aa_att, bb_att, pp_att, xx_ffn )

		if len(ctx) == 0:
			char = sample_logits( x.tolist() )
			char = char.item()
			emb = model.w.emb.weight[char]

			print(tokenizer.decode(char), end='', flush=True)

	print("")



def rnn_export():
	model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)

	emb = torch.rand(n_embd)
	xx_att = torch.zeros(n_layer, n_embd)
	aa_att = torch.zeros(n_layer, n_embd)
	bb_att = torch.zeros(n_layer, n_embd)
	pp_att = torch.zeros(n_layer, n_embd) - 1e30
	xx_ffn = torch.zeros(n_layer, n_embd)

	model.w.emb.weight.numpy().tofile("emb.weight.bin")
	torch.onnx.export(model, args=(emb, xx_att, aa_att, bb_att, pp_att, xx_ffn), f="rwkv.onnx", input_names = ["emb", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"], output_names = ["x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"], verbose=False)

