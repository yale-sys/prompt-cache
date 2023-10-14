.PHONY: eval

LLM_CONFIG_PATH ?= ./config/llm_config_llama2.json
# LLM_CONFIG_PATH ?= ./config/llm_config_llama2_13b.json
# DATASET ?= squad_v2
# DATASET ?= multi_news
# DATASET ?= ms_marco
DATASET ?= narrativeqa
ENABLE_CACHE ?= False
SPLIT ?= 0,1
TEST_LATENCY ?= False
USE_CPU_FOR_INFERENCE ?= False
eval:
	python3 eval.py \
		--llm_config_path $(LLM_CONFIG_PATH) \
		--dataset $(DATASET) \
		--enable_cache $(ENABLE_CACHE) \
		--split $(SPLIT) \
		--test_latency= $(TEST_LATENCY) \
		--use_cpu_for_inference $(USE_CPU_FOR_INFERENCE) \
		--cache_batch_size 1
