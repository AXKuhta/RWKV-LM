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

	xx_att = torch.zeros(12, 768)
	aa_att = torch.zeros(12, 768)
	bb_att = torch.zeros(12, 768)
	pp_att = torch.zeros(12, 768) - 1e30
	xx_ffn = torch.zeros(12, 768)

	ctx = tokenizer.encode(context)

	for i in range(64):
		ttx = [] + ctx

		while len(ttx) < 768:
			ttx.insert(0, 0)

		x, xx_att, aa_att, bb_att, pp_att, xx_ffn  = model( torch.tensor(ttx), xx_att, aa_att, bb_att, pp_att, xx_ffn )

		char = sample_logits( x.tolist() )
		char = char.item()

		print(tokenizer.decode(char), end='', flush=True)
		ctx.append(char)

def rnn_export():
	model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)

	ctx = torch.randint(5000, (1024,), dtype=torch.int32 ) + 100
	xx_att = torch.zeros(12, 768)
	aa_att = torch.zeros(12, 768)
	bb_att = torch.zeros(12, 768)
	pp_att = torch.zeros(12, 768) - 1e30
	xx_ffn = torch.zeros(12, 768)

	torch.onnx.export(model, args=(ctx, xx_att, aa_att, bb_att, pp_att, xx_ffn), f="rwkv.onnx", input_names = ["idx", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"], output_names = ["x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"], verbose=True)
