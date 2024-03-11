.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sat2plan || :
	@pip install -e .

run_train_unet:
	python3 -c 'from sat2plan.interface.main import train_unet; train_unet()'

run_train_dcgan:
	python3 -c 'from sat2plan.interface.main import train_dcgan; train_dcgan()'
