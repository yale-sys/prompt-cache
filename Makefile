.PHONY: eval

# LLM_CONFIG_PATH ?= ./config/llm_config_falcon.json
LLM_CONFIG_PATH ?= ./config/llm_config_llama2.json
DATASET ?= squad_v2
# DATASET ?= multi_news
# DATASET ?= ms_marco
# DATASET ?= narrativeqa
ENABLE_CACHE ?= False
SPLIT ?= 0,1
TEST_LATENCY ?= False
USE_CPU_FOR_INFERENCE ?= False
eval:
	CUDA_VISIBLE_DEVICES=0 python3 eval.py \
		--llm_config_path $(LLM_CONFIG_PATH) \
		--dataset $(DATASET) \
		--enable_cache $(ENABLE_CACHE) \
		--split $(SPLIT) \
		--test_latency $(TEST_LATENCY) \
		--use_cpu_for_inference $(USE_CPU_FOR_INFERENCE) \
		--verbose False
