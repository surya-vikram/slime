# Chimera 10B architecture. YaRN metadata is read from the HF checkpoint by
# slime_plugins.models.chimera.model_provider.

NLAYERS=25
FIRST_K_DENSE_REPLACE=2

arr=()
for ((i = 0; i < NLAYERS; i++)); do
    if ((i < FIRST_K_DENSE_REPLACE)); then
        arr+=(0)
    else
        arr+=(1)
    fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=,; echo "${arr[*]}")"

MODEL_ARGS=(
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 2
    --kv-channels 256
    --num-layers 25
    --hidden-size 2048
    --ffn-hidden-size 8192

    --normalization RMSNorm
    # The image parser accepts RoPE here; the custom provider replaces it
    # with the HF-authoritative Chimera YaRN configuration before model build.
    --position-embedding-type rope
    --norm-epsilon 1e-5
    --rotary-percent 1.0
    --rotary-base 10000000
    --no-rope-fusion
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --vocab-size 50176

    --num-experts 32
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size 2048
    --moe-router-topk 4
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-router-bias-update-rate 0.0
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0.0
    --moe-z-loss-coeff 0.001
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
)
