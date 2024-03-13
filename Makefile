.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sat2plan || :
	@pip install -e .

run_train_unet:
	python3 -c 'from sat2plan.interface.main import train_unet; train_unet()'

run_train_ucvgan:
	python3 -c 'from sat2plan.interface.main import train_ucvgan; train_ucvgan()'

run_train_sam_gan:
	python3 -c 'from sat2plan.interface.main import train_sam_gan; train_sam_gan()'

run_api:
	python3 -c 'from sat2plan.api.api import test_api; test_api()'
