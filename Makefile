.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sat2plan || :
	@pip install -e .

run_preprocess:
	python -c 'from sat2plan.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from sat2plan.interface.main import train; train()'

run_pred:
	python -c 'from sat2plan.interface.main import pred; pred()'

run_evaluate:
	python -c 'from sat2plan.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

