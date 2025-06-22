from math import sqrt
import torch
import torch.nn
import torch.nn.functional as F

class ColumnParallelLinear(Linear):
	"""
	Linear layer with column parallelism, splitting features across distributed processes

	Args:
		in_features (int): Number of input features.
		out_features (int): Number of output features.
		bias (bool): Whether to include a bias term. Default to False.
		dtype (optional): Data type for the layer. Defaults to 'torch.bfloat16'
	"""
	def __init__(self, in_features:int, out_features:int, bias:tool = False, dtype = None):
		assert out_features % world_size == 0, f"Output features must be divisible by world size for parallelism"
		self.part_out_features = output_features // world_size
		super().__init__(in_features, self.part_out_features, bias, dtype)

	def forward(self, x:torch.Tensor) ->torch.Tensor:
		"""
		Forward pass for column parallel linear layer.

		Args:
			x (torch.Tensor): Input tensor.
		Returns:
			torch.Tensor: Transformed tensor with column-parallel computation
		"""

		y = linear(x, self.weight, self.bias)

		return y
class MLA(nn.Module):
	"""
	qk_head_dim (int): Total dimensionality
	v_head_dim (int): Dimensionality of value projections.
	softmax_scale (float): Scaling factor for softmax in attention computation.
	"""
	def __init__(self, args:ModelArgs):
		super().__init__()
		self.dim = args.dim
		self.n_heads = args.n_heads
		self.n_local_heads = args.n_heads
		self.q_lora_rank = args.q_lora_rank
		self.kv_lora_rank = args.kv_lora_rank
		self.qk_nope_head_dim = args.qk_nope_head_dim
		self.qk_rope_head_dim = args.qk_nope_head_dim
		self.qk_head_dim = args.qk_head_dim
		self.v_head_dim = args.v_head_dim

		if self.q_lora_rank == 0:
			self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
		else:
			## wq_a: first part of generating matrix, converting token vector to latent vector
			self.wq_a = Linear(self.dim, self.q_lora_rank)
			## normalize to not mess up our matrix
			self.q_norm = RMSNorm(self.q_lora_rank)
			self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
		self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_head_dim)
		self.kv_norm = RMWSNorm(self.kv_lora_rank)
		self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads)
		self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
		self.softmax_scale = self.qk_head_dim ** -0.5
		## rope: relative position coding
		## no additional parameters
		## allows model to handle longer seq than they are trained on
		## softmax: reverse of temperature, increase softmax scale, less randomness
		if args.max_seq_len > args.original_seq_len:
			mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
			self.softmac_scale = se;f.softmax_scale * mscale * mscale
		if attn_impl == "naive":
			self.register_buffer["k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim, persistent = False)]
			self.register_buffer["v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim, persistent = False)]
			self.register_buffer["kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, persistent = False)]
			self.register_buffer["pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_lora_rank, persistent = False)]
	def forward(self, x:torch.Tensor, start_pos: int, freq_cis:torch.Tensor, mask:Optional[torch.Tensor]):
		"""
		Forward pass for the Multi-Headed Attention Layer (MLA).

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
			start_pos (int): Starting position in the sequence for caching.
			freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings
			mask (Optional[torch.Tensor]): Mask tensor to exclude certain position
		Returns:
			torch.Tensor: Output tensor with the same shape as the input
		"""
		bsz, seqlen = x.size() ## used to reshape the tensor
		end_pos = start_pos + seqlen 
		if self.q_lora_rank == 0:
			q = self.wq(x)
		else:
			## c^Q_t = W^D*Qh_t
			## q^C_t = W^UQ * c_t^Q
			q = self.wq_b((self.q_norm(self.wq_a(x))))
		q = q.view(bsz, seqlen, self.n_local_heads,self.qk_head_dim)
		q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = 1)
		q_pe = apply_rotary_emb([q_pe, freqs_cis])
		kv = self.wkv_a(x)
		kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim =-1)
		k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
		if attn_impl == "naive":
			q = torch.cat([q_nope, q_pe] dim = -1)
			kv = self.wkv_b(self.kv_norm(kv))
			self.k_cache[:bsz, start_pos:end_pos] = k 
			self.v_cache[:bsz, start_pos:end_pos] = v 
			scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
		else: ##deepseek implementation
			wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight.dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
			wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
			q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:,:self.qk_nope_head_dim])
			self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
			self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
			scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])) + torch.einsum("bshr,btr->bsht",q_pe, self.pe_cache[:bsz,:end_pos]) * self.softmax_scale
		if mask is not None:
			scores += mask.unsqueeze(1)
		scores = scores.softmax(dim = -1, dtype = torch.float32).type_as(x)
		if attn_impl == "naive":
			x = torch.einsum("bsht,bthd->bshd", scores,self.v_cahe[:bsz, :end_pos])
		else:
			x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz:end_pos])
			x = torch.einsum("bshc,hdc->bshd",x, wkv_b[:,-self.v_head_dim])
		return x

class Gate(nn.Module):
	def forward:
		if self.score_func == "softmax":
			scores = scores.softmax(dim = -1, dtype = torch.float32)
		else:
			scores = scores.sigmoid()
		original_scores = scores
		if self.bias is not None:
			scores += self.bias
		if self.n_groups > 1:
			scores = scores.view(x.size(0), self.n_groups, -1)
			if self.bias is None:
				group_scores = scores.amax(dim=-1)
			else:
				group_scores = scores.topk(2, dim = -1)[0].sum(dim = -1)
			indices = group_scores.topk(self.topk_groups, dim = -1)[1]
			mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter(1,indices, False)
			scores.mask_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
		indices = torch.topk(scores, self.topk, dim = -1)[1]
		weights = original_scores.gather(1, indices)

class MLP(nn.Module):
	"""
	Multilayer Perceptron used as a feed-forward layer
	Attributes: 
	    w1.nn.Module): Linear layer for input-to-hidden transformation
	    w2.nn.Module): Linear layer for hidden-to-output transformation
	    w3.nn.Module): Additional linear layer for feature transformation
	"""
	def __init__(self, dim:int, inter_dim:int):
		"""
		Initializes the MLP layer.
		Args:
		    dim(int): input and output dimensionality
		    inter_dim (int): Hidden layer dimensionality
		"""
		super().__init__()
		self.w1 = ColumnParallelLinear(dim, inter_dim)
		self.w2 = RowParallelLinear(inter_dim, dim)
		self.w3 = ColumnParallelLinear(dim, inter_dim)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		return self.w3(F.silu(self.w1(x) * self.w3(x)))

class MOE(nn.Module):
	def __init__(self, args:ModelArgs):
		"""
		Args:
		 args (ModelArgs): Model arguments containing MOE parameters
		"""
		self.dim = args.dim 
		assert args.n_routed_experts % world.size == 0 f"Number of experts must be divisible by world size {world_size}"""
		self.n_routed_experts = args.n_routed_experts
		self.n_local_experts = args.n_routed_experts // world_size
		self.n_activated_experts = args.n_activated_experts
		self.experts_start_idx = rank * self.n_local_experts
		self.experts_end_idx = self.experts_start_idx + self.n_local_experts
		self.gate = Gate(args)
		self.experts = nn.ModuleList ([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None for i in range(self.n_routed_experts)] )
		## GPU 1: expert ,None, None...
		## GPU 2: None, expert, expert
		## ...
		##
		self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

	def forward(self,x:torch.Tensor) -> torch.Tensor:
		shape = x.size()
		x = x.view(-1, self.dim)
		weights, indices = self.gate(0)
		y = torch.zeros_like(x)
		counts = torch.bincount(indices.flatten(), minlength = self.n_routed_experts).to_list()
		for i in range(self.experts_start_idx, self.experts_end_idx):
			## iterate each expert assigned to this gpu node
			if counts[i] == 0:
				continue
			idx, top = torch.where(indices == i)
			y[idx] += expert(x[idx]) * weights[idx, top, None]
		z = self.shared_experts(x)
		if world.size > 1:
			dist.all_reduce(y)
		return (y+z).view(shape)

class Block(nn.Module):
	def __init__(self, layer_id: int, args:ModelArgs):
		"""
		Initializing the Transformer block
		Args:
	    	layer_id: Layer idx in the transformer
	    	args(ModelArgs : Model arguments containing block parameters
		"""
		super().__init__()
		self.attn = MLA(args)
		self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MOE(args)
		self.attn_norm = RMSNorm(args.dim)
		self.ffn_norm = RMSNorm(args.dim)

	def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]) -> torch.Tensor:
		x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)

class ModelArgs:
	"""
	Attributes: ...
	beta_slow slow rotation
	beta_fast fast rotation
	"""
	max_batch_size:int = 8
	max_seq_len: int = 4096*4
	dtype: Literal["bf16","fp8"] = "bf16"
	dim:int = 2048
	vocab_size:int = 10944
	inter_dim:int = 10944
	moe_inter_dim: int = 1408
	n_layers:int = 27
	n_dense_layers: int = 1
	n_heads:int = 16
	## moe
	n_routed_experts:int = 64
	n_shared_experts:int = 2
	n_activated_experts: int = 2
	n_expert_groups: int = 1
	n_limited_grousp: int = 1
	score_func: Literal["softmax", "sigmoid"] = "softmax"
	route_scale: float = 1
	#mla
	q_lora_rank: int = 0
	kv_lora_rank: int = 512
	qk_nope_head_dim: int = 128
	qk_rope_head_dim:int = 64
	v_head_dim:int = 128
	# yarn
	original_seq_len:int = 4096
	rope_theta: float = 10000.0 
	rope_factor: float = 40 

class Transformer(nn.Module):
	def __init__(self, args:ModelArgs):
		global world_size, rank 
		world_size = dist.get_world_size() if dist.is_initialized() else 1
		rank = dist.get_rank() if dist.is_initialized() else 0
		Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
		super().__init__()
		self.max_seq_len = args.max_seq_len
		self.embed = ParallelEmbedding(args.vocab_size, args.dim)
		self.layers = torch.nn.ModuleList()
		for layer_id in range(args.n_layers()):
			self.layer.append(Block(layer_id, args))
		self.norm = RMSNorm(args.dim)
		self.head = ColumnParallelLinear(args.dim ,args.vocab_size,dtype=torch.get_default_dtype())
		self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent =False)

	@torch.inference_mode()
	def forward(self, tokens:torch.Tensor, start_pos: int = 0):

		seqlen = tokens.size(1)
		h = self.embed(tokens)
		freqs_cis = self.freq_cis[start_pos: start_pos+seqlen]
		mask = None 
		if seqlen > 1:
			mask = torch.full((seqlen, seqlen),float("-inf"),device = tokens.device).triu_(1)
		for layer in self.layers:
			h = layer(h, start_pos, freq_cis,mask)
		h = self.norm(h)[:,-1]
		logits = self.head(h):
		if world_size > 1:
			all_logits = [torch.empty_like(logits) for _ in range (world_size)]
			dist.allgather(all_logits, logits)
			logtis = torch.cat(all_logits, dim = -1)
		logits = logits.reshape(batch_size, seqlen, -1)
		loss = None 
		if targets is not None:
			shift_logits = logits[:,:-1,:].contiguous()
			shift_targets = targets[:,1:].contiguous()
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(shit_logits.view(-1, shit_logits.size(-1)),shift_targets.view(-1))
		return logits, loss

if __name__ == "__main__":
	torch.set_default_dtype(torch.bfloat16)
	torch.set_default_device("cuda")
	torch.manual_seed(0)
	args = ModelArgs()
	x = torch.randint(0, args.vocab_size, (2,128))
	model = Transformer(args)
	print(model(x).size())
















class RowParallelism: 




		