
# setup env
Create env using conda and environnment.yml
Install pip requirements.txt
Then pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

# Start Visdom server

python -m visdom.server

# Train

python train.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction AtoB

# Test

python test.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction AtoB

NB :
- add "--preprocess true" to use the prepross model
- aad "--tevnet_weights <path>" to use tevnet and precise "--lambda_tevnet XX" default is 15.0

# Run scoring
python result_metrics.py --experiment <Name of the experiment> ex: python result_metrics.py --experiment KAIST_IRGAN_preprocess_v3

# Acknowledgments

Our code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , https://github.com/NVIDIA/pix2pixHD, https://github.com/facebookresearch/ConvNeXt and https://github.com/CXH-Research/IRFormer.


