# Final Chimera 10B YaRN-on-RoPE architecture. YaRN metadata is read from the
# HF checkpoint by slime_plugins.models.chimera.model_provider.

NLAYERS=25
FIRST_K_DENSE_REPLACE=2
CHIMERA_HIDDEN=2048
CHIMERA_FFN=8192
CHIMERA_HEADS=16
CHIMERA_HEAD_DIM=256
CHIMERA_EXPERTS=32
CHIMERA_TOPK=4
CHIMERA_EXPERT_FFN=2048
case "${CHIMERA_MODEL_SIZE:-full}" in
    full) ;;
    tiny)
        # Exactly Megatron-LM/examples/chimera/tiny_chimera.sh geometry.
        NLAYERS=8
        CHIMERA_HIDDEN=512
        CHIMERA_FFN=2048
        CHIMERA_HEADS=8
        CHIMERA_HEAD_DIM=64
        CHIMERA_EXPERTS=8
        CHIMERA_TOPK=2
        CHIMERA_EXPERT_FFN=256
        ;;
    *) echo 'Invalid CHIMERA_MODEL_SIZE' >&2; return 1 ;;
esac

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
    --num-attention-heads "$CHIMERA_HEADS"
    --num-query-groups 2
    --kv-channels "$CHIMERA_HEAD_DIM"
    --num-layers "$NLAYERS"
    --hidden-size "$CHIMERA_HIDDEN"
    --ffn-hidden-size "$CHIMERA_FFN"

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

    --num-experts "$CHIMERA_EXPERTS"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size "$CHIMERA_EXPERT_FFN"
    --moe-router-topk "$CHIMERA_TOPK"
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
