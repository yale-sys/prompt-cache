.PHONY: eval

LLM_CONFIG_PATH ?= ./benchmark/config/llm_config_llama2.json
DATASET ?= squad_v2
ENABLE_CACHE ?= True

eval:
	python3 ./benchmark/eval.py \
		--llm_config_path $(LLM_CONFIG_PATH) \
		--dataset $(DATASET) \
		--enable_cache $(ENABLE_CACHE)