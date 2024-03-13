.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y sat2plan || :
	@pip install -e .

run_train_unet:
	python3 -c 'from sat2plan.interface.main import train_unet; train_unet()'

<<<<<<< HEAD
run_train_dcgan:
	python3 -c 'from sat2plan.interface.main import train_dcgan; train_dcgan()'
=======
run_train_sam_gan:
	python3 -c 'from sat2plan.interface.main import train_sam_gan; train_sam_gan()'
>>>>>>> 605a57e1026dadba72c2909bea7ed61a5f8db64f
