CUDA_VISIBLE_DEVICES=0 \
python test.py \
--weights-file /w/340/murdock/irgan/tev-out/KAIST/epoch_60.pth \
--image-dir /w/340/murdock/irgan/datasets/KAIST/test \
--smp_model Unet \
--smp_encoder resnet18 \
--output-dir /w/340/murdock/irgan/tev-out/images \
--vnums 4