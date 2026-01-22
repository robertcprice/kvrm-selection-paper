.PHONY: help reproduce-all reproduce-tile-selection reproduce-regret reproduce-hierarchical reproduce-latency clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-30s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

reproduce-all: ## Run all experiments
	@echo "Running all KVRM experiments..."
	python scripts/run_tile_selection.py
	python scripts/run_regret_analysis.py
	python scripts/run_hierarchical_experiments.py
	python scripts/run_latency_analysis.py
	@echo "All experiments complete!"

reproduce-tile-selection: ## Run tile selection experiments
	@echo "Running tile selection experiments..."
	python scripts/run_tile_selection.py

reproduce-regret: ## Run regret analysis
	@echo "Running regret analysis..."
	python scripts/run_regret_analysis.py

reproduce-hierarchical: ## Run hierarchical registry experiments
	@echo "Running hierarchical registry experiments..."
	python scripts/run_hierarchical_experiments.py

reproduce-latency: ## Run tail latency analysis
	@echo "Running tail latency analysis..."
	python scripts/run_latency_analysis.py

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	@echo "Clean complete!"
