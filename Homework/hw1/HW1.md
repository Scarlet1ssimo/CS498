# Assignment 1

Muchen Xu

## Problem 1

### Q1

<!-- (1 points) Given a transformer with a model hidden dimension of 768 with 12 attention heads,
what is the head hidden dimension? -->

As prediction of each heads will be concatenated together, the head hidden dimension = 768/12=64.

### Q2

<!-- Q2. (1 points) Assume that we have a sequence length of s tokens, what is the per-token time-complexity of self-attention using the KV-cache? -->

The $i$-th token will only attend to the previous $i-1$ tokens, leading to time complexity of $O(i)$. Sum over $s$ tokens, we have total time complexity of $O(s^2)$. Therefore, the ammortized, per-token time-complexity is $O(s)$.

### Q3


<!-- Q3. (2 points) During the training of Deepseek V3, what data-type are the optimizer master weights stored in? -->

The optimizer master weights are stored in fp32. 

> However, the *master weights (stored by the optimizer)* and gradients (used for batch size accumulation) *are still retained in FP32* to ensure numerical stability throughout training. (DeepSeek V3 paper)

### Q4
<!-- Q4. (2 points) Given a transformer with L = 61 layers, h = 128 heads per layer, and dk = 128 head
dimension, what is the total number of kv cache entries per token when using MHA and MQA. -->

MHA: Inside a layer, each token is divided into $128$ heads. In each head, there are $2$ vectors (key and value) of dimension $128$. Therefore, the total number of kv cache entries per token is $128*2*128*61=1998848$.

MQA: Inside a layer, $128$ heads share the same set of key and value pairs. In each head, there are still $2$ vectors (key and value) of dimension $128$. Therefore, the total number of kv cache entries per token is $2*128*61=15616$.

### Q5

<!-- Q5. (4 points) Given a standard MHA architecture with L = 64 layers, dmodel = 8192 model hidden
dimension, h = 32 heads per layer, dk = 256 head dimension, and ntokens = 65536, what is
the total number of parameters and memory required to store the weights in bf16, assuming
an MLP up-down projection factor of 4? Exclude biases, layernorm, and positional encoding
parameters. Focus primarily on the token embeddings, attention, and MLP modules. -->

Embedding layer: $65536*8192=536870912$ parameters.

Parameters of MHA per layer includes $W_Q, W_K, W_V, W_O$: $4*8192*8192=268435456$

Parameters of MLP per layer: $8192*4*8192 + 4*8192*8192=536870912$

So the total number of parameters will be:

$$
65536*8192+64*(4*8192*8192 + 8192*4*8192 + 4*8192*8192)=52076478464 
$$

So there are $52076478464$ parameters. And the memory required to store the weights in bf16 is: $52076478464 * 2$ bytes = $104152956928$ bytes ≈ 97.0 GiB.
