.PHONY: eval

LLM_CONFIG_PATH ?= ./config/llm_config_llama2.json
DATASET ?= squad_v2
# DATASET ?= multi_news
ENABLE_CACHE ?= True
SPLIT ?= 0,1

eval:
	python3 eval.py \
		--llm_config_path $(LLM_CONFIG_PATH) \
		--dataset $(DATASET) \
		--enable_cache $(ENABLE_CACHE) \
		--split $(SPLIT)