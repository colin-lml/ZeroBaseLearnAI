VOCAB_SIZE=500    # 词表大小
MAX_SEQ_LEN=2000     # GPT模型输入限制

# transformer
GPT_DIM=512
GPT_HEAD=8
GPT_FF=1024
GPT_BLOCKS=2

# training
TRAIN_ITER=1000
BATCH_SIZE=50

# inference
TEMPERATURE=1.2
TOP_K=20

# special tokens
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
PAD='<|padding|>'
IM_START='<|im_start|>'
IM_END='<|im_end|>'

# chat or generate
GPT_MODE='generate'