# The Surprising Effectiveness of Representation Learning for Visual Imitation
**Authors**: Jyothish Pari*, Nur Muhammad (Mahi) Shafiullah*, Sridhar Pandian Arunachalam, [Lerrel Pinto](https://lerrelpinto.com)

This is an original PyTorch implementation of Visual Imitation through Nearest Neighbors, or VINN from the paper [The Surprising Effectiveness of Representation Learning for Visual Imitation](https://jyopari.github.io/VINN/)


## Execution on Real Robot
 <p align="center">
  <img width="45%" src="https://jyopari.github.io/VINN/files/vinn1.gif">
  <img width="45%" src="https://jyopari.github.io/VINN/files/vinn2.gif">
 </p>

## Method
![VINN](https://jyopari.github.io/VINN/files/method.png)
During training, we use offline visual data to train a BYOL-style self-supervised model as our encoder. During evaluation, we compare the encoded input against the encodings of our demonstration frames to find the nearest examples to our query. Then, our model's predicted action is just a weighted average of the associated actions from the nearest images.

## Setup
The following libraries are required for running our offline code:
* [PyTorch](https://pytorch.org/) with torchvision and other dependencies.
* [byol-pytorch](https://github.com/lucidrains/byol-pytorch)

You can install them with `pip`.
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install byol-pytorch
```

## Dataset

### Demonstration Collection
<p align="center">
  <img width="45%" src="https://jyopari.github.io/VINN/files/demo.gif">
  <img width="45%" src="https://jyopari.github.io/VINN/files/demo_pov.gif">
</p>

### Downloadable dataset
All our data can be found at this url: [https://drive.google.com/drive/folders/11-sAN2a-F7G-lvx6qRXnrWjlxNb0PH1m](https://drive.google.com/drive/folders/11-sAN2a-F7G-lvx6qRXnrWjlxNb0PH1m).

## Example Commands

<details><summary>BYOL - Handle Datset/Door Opening</summary></summary>

```
python representation_models/BYOL.py	--batch_size 168 \
					--root_dir /path/to/non_parametric_imitation/ \
					--folder /path/to/train_dir/ \
					--dataset HandleData \
					--extension handle \
					--img_size 224 \
					--epochs 101 \
					--wandb 1 \
					--gpu 1 \
					--hidden_layer avgpool \
					--pretrained 1 \
					--save_dir /path/to/chkpts/
```
</details>
<details><summary>BC on Representations - Handle Dataset/Door Opening</summary>

```
python train_BC.py	--bc_model BC_rep \
			--t 0 \
			--batch_size 256 \
			--root_dir /path/to/non_parametric_imitation/ \
       			--gpu 1 \
			--img_size 224 \
			--train_dir /path/to/train_dir/ \
			--val_dir /path/to/val_dir/ \
			--test_dir /path/to/test_dir/ \
			--dataset HandleData \
			--representation_model_path /path/to/chkpts/ssl_handle_model \
			--model BYOL \
			--layer avgpool \
			--architecture ResNet \
			--eval 0 \
			--temporal 0 \
			--wandb 1 \
			--lr 0.001 \
       			--epochs 8001 \
			--pretrained 1 \
			--save_dir /path/to/chkpts/
```
</details>
<details><summary>BC End to End - Handle Dataset/Door Opening</summary>

```
python train_BC.py	--bc_model BC_end_to_end \
			--t 0 \
			--batch_size 64 \
			--root_dir /path/to/non_parametric_imitation/ \
			--gpu 1 \
			--img_size 224 \
			--train_dir /path/to/train_dir/ \
			--val_dir /path/to/val_dir/ \
			--test_dir /path/to/test_dir/ \
			--dataset HandleData \
			--representation_model_path None \
			--model None \
			--layer avgpool \
			--architecture ResNet \
			--eval 0 \
			--temporal 0 \
			--wandb 1 \
			--lr 0.001 \
			--epochs 101 \
			--pretrain_encoder 1 \
			--save_dir /path/to/chkpts/
```
</details>
<details><summary>BYOL - Push/Stack Dataset</summary>

```
python representation_models/BYOL.py	--batch_size 168 \
					--root_dir /path/to/non_parametric_imitation/ \
					--folder /path/to/train_dir/ \
					--dataset PushDataset \
					--extension push \
					--img_size 224 \
					--epochs 101 \
					--wandb 1 \
					--gpu 1 \
					--hidden_layer avgpool \
					--pretrained 1 \
					--save_dir /path/to/chkpts/
```
</details>
<details><summary>BC on Representations - Push/Stack Dataset</summary>

```
python train_BC.py	--bc_model BC_rep \
			--t 0 \
			--batch_size 256 \
			--root_dir /path/to/non_parametric_imitation/ \
       			--gpu 1 \
			--img_size 224 \
			--train_dir /path/to/train_dir/ \
			--val_dir /path/to/val_dir/ \
			--test_dir /path/to/test_dir/ \
			--dataset PushDataset \
			--representation_model_path /path/to/chkpts/ssl_push_model \
			--model BYOL \
			--layer avgpool \
			--architecture ResNet \
			--eval 0 \
			--temporal 0 \
			--wandb 1 \
			--lr 0.001 \
       			--epochs 8001 \
			--pretrained 1 \
			--save_dir /path/to/chkpts/
```
</details>
<details><summary>BC End to End - Push/Stack Dataset</summary>

```
python train_BC.py	--bc_model BC_end_to_end \
			--t 0 \
			--batch_size 64 \
			--root_dir /path/to/non_parametric_imitation/ \
			--gpu 1 \
			--img_size 224 \
			--train_dir /path/to/train_dir/ \
			--val_dir /path/to/val_dir/ \
			--test_dir /path/to/test_dir/ \
			--dataset PushDataset \
			--representation_model_path None \
			--model None \
			--layer avgpool \
			--architecture ResNet \
			--eval 0 \
			--temporal 0 \
			--wandb 1 \
			--lr 0.001 \
			--epochs 101 \
			--pretrain_encoder 1 \
			--save_dir /path/to/chkpts/
```
</details>

If you use this code in your research project please cite us as: 
```
@misc{VINN,
  author = {Pari, Jyo and Shafiullah, Mahi and Arunachalam, Sridhar and Pinto, Lerrel},
  title = {Visual Imitation through Nearest Neighbors (VINN) implementation},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jyopari/VINN/tree/main}},
}
```

