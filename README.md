#  Towards Vivid and Diverse Image Colorization with Generative Color Prior, ICCV 2021

> [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2108.08826)<br>
> [Yanze Wu](https://github.com/ToTheBeginning), [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Xun Zhao](https://scholar.google.com.hk/citations?user=KF-uZFYAAAAJ&hl=en), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)<br>
> [ARC Lab, Tencent PCG](https://arc.tencent.com/en/index)

<p align="center">
  <img src="assets/teaser.png" width="800px">
</p>

---

## Installation
### Core dependencies
* CUDA >= 10.0 (test on 10.0, 10.2, and 11.1)
* gcc >= 7.3
* pytorch >= 1.6 (test on 1.6, 1.7.1, and 1.9.0)
* python3 (test on 3.8)
* yacs

### Install DCN package (only required by torchvision < 0.9.0)
```shell
# you can skip this step if you are using torchvision >= 0.9.0
cd ops
python setup.py develop
```

## Download the Pre-trained Models

Download the pretrained models from [[Google Drive](https://drive.google.com/drive/folders/1-mwYyvF4nGbqI3x1dC-OruX0tru02JKE?usp=sharing) ] and put them into the `assets` folder.
If you want to reproduce or get the results reported in our ICCV 2021 paper for academic purpose, you can check [model zoo](model_zoo.md).
Otherwise, you just need to use the default options, which is our best model.

## Inference
### Test in the wild images
1. Predict ImageNet label (0-999) for these images
   1. install awesome timm, `pip install timm`
   2. use a SOTA classification model from timm to predict the labels
      ```shell
      python predict_imagenet_label.py testcase_in_the_wild --model beit_large_patch16_512 --pretrained
      ```
      here `testcase_in_the_wild` folder has the images you want to test
   3. you will get the label map in `assets/predicted_label_for_user_image.txt`

2. Inference colorization
    ```shell
    python main.py --expname inference_in_the_wild --test_folder testcase_in_the_wild DATA.FULL_RES_OUTPUT True
    ```
    ```Console
    options:
        --expname: the results will be saved in results/{expname}-{time} folder
        --inference_in_the_wild: contains the images you want to test
        --bs: batch size
        DATA.FULL_RES_OUTPUT: True or False. If set to True
                              the full resolution results will be saved in results/{expname}-{time}/full_resolution_results
                              batch size should be 1 if this flag is set to True
        DATA.CENTER_CROP: whether to center crop the input images.
                          This flag and DATA.FULL_RES_OUTPUT flag can not be both set to True
    ```
3. If everything goes well, you will get the results similar as [visual_results.md](visual_results_in_the_wild.md).


### Test images from ImageNet val set
* The most worry-free way is to make sure the images' names are **consistent** with the official ImageNet name.
Because we provide the GT ImageNet labels in [imagenet_val_label_map.txt](assets/imagenet_val_label_map.txt).
You can check `testcase_imagenet` for examples.
* Also, the test images should better be **color** images rather than grayscale images. If you want to test on grayscale images, please read the `read_data_from_dataiter` function from [base_solver.py](solvers/base_solver.py),
prepare the grayscale images following the pipeline and hack the related code.
* Inference
    ```shell
    python main.py --expname inference_imagenet --test_folder testcase_imagenet DATA.FULL_RES_OUTPUT False DATA.CENTER_CROP False
    ```
* If everything goes well, you will get the following quantitative results on the full 50,000 ImageNet validation images (yes, FID is better than the number reported in our ICCV 2021 paper)

  | eval@256x256 | FID↓ | Colorfulness↑ | ΔColorfulness↓ |
  | :-----| ----: | :----: | :----: |
  | w/o center crop | 1.325 | 34.89 | 3.45 |
  | w/ center crop | 1.262 | 34.74 | 4.12 |


### Diverse Colorization
You can achieve **diverse colorization** by:
1. **adding random noise to the latent code**
   ```shell
   python main.py --expname inference_random_diverse_color --latent_direction -1 --test_folder testcase_in_the_wild/testcase19_from_eccv2022_bigcolor.png DATA.FULL_RES_OUTPUT True
   ```
   | Input | Diverse colorization | Diverse colorization | Diverse colorization |
   | :----: | :----: | :----: | :----: |
   | <img src="testcase_in_the_wild/testcase19_from_eccv2022_bigcolor.png" width="224"> |<img src="https://user-images.githubusercontent.com/11482921/196039151-fca6d31a-63f2-4194-b2c0-827296ad090c.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039209-68d0064a-52c5-4c83-8799-41dc68f50378.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039232-e8726c4e-b9fa-4430-8c94-002d3d9f41f9.png" width="224"> |
2. **walking through the interpretable latent space**

   we use the method described in `Unsupervised Discovery of Interpretable Directions in the GAN Latent Space` to find the interpretable directions for BigGAN.
   By setting `latent_direction` to the color-relevant direction (_e.g._, 4, 6, 23) we found, you can achieve more controllable diverse colorization.

   ps: we also provide the checkpoint (i.e., the full direction matrix), so you can find more color-relevant directions by yourself.
   ```shell
   # get the label first
   python predict_imagenet_label.py testcase_diverse --model beit_large_patch16_512 --pretrained
   # the 1st direction we found
   python main.py --expname inference_diverse_color_dir4 --latent_direction 4 --test_folder testcase_diverse/test_diverse_inp0.png
   # the 2nd direction we found
   python main.py --expname inference_diverse_color_dir6 --latent_direction 6 --test_folder testcase_diverse/test_diverse_inp1.png
   python main.py --expname inference_diverse_color_dir6 --latent_direction 6 --test_folder testcase_diverse/test_diverse_inp2.png
   # the 3rd direction we found
   python main.py --expname inference_diverse_color_dir23 --latent_direction 23 --test_folder testcase_diverse/test_diverse_inp3.png
   ```
   | Input | Diverse colorization | Diverse colorization | Diverse colorization |
   | :----: | :----: | :----: | :----: |
   | <img src="testcase_diverse/test_diverse_inp0.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039397-a4cb0a0e-0825-4e5e-ba9c-774db1c66962.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039399-d325f26e-905b-4171-9355-b1cbfa2c2cc5.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039402-727530b1-d445-4482-b53e-a3f71fb4a206.png" width="224"> |
   | <img src="testcase_diverse/test_diverse_inp1.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039508-d0d31afb-181c-4e82-867a-91da3175a626.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039512-7e4fbc38-76ca-4639-9d93-7ba0d37ec129.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039516-fd6ca5b5-efb3-4fed-bf46-20f8a17caa88.png" width="224"> |
   | <img src="testcase_diverse/test_diverse_inp2.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039603-ce380983-493b-467b-bdd0-13cbd67086d4.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039605-6708f75e-1c1b-448c-83e4-9d94c3c10679.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039607-f7d221ee-2041-4e43-b16c-b2673b8443d0.png" width="224"> |
   | <img src="testcase_diverse/test_diverse_inp3.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039652-210287ad-3cf8-46b2-ac0f-0d0415f4c749.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039657-43ec1bbb-3584-4c14-b386-01029bedae5a.png" width="224"> | <img src="https://user-images.githubusercontent.com/11482921/196039658-b54adee0-2997-442e-a219-6abd34377852.png" width="224"> |
3. **changing the category**

   you can modify the `assets/predicted_label_for_user_image.txt`


### TODO
- [ ] add colab demo

## Citation
If you find this project useful for your research, please consider citing our paper:
```bibtex
@inproceedings{wu2021towards,
  title={Towards vivid and diverse image colorization with generative color prior},
  author={Wu, Yanze and Wang, Xintao and Li, Yu and Zhang, Honglun and Zhao, Xun and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Acknowedgement
This project borrows codes from [CoCosNet](https://github.com/microsoft/CoCosNet), [DGP](https://github.com/XingangPan/deep-generative-prior), and [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
DCN code from early version of [mmcv](https://github.com/open-mmlab/mmcv). predict_imagenet_label.py from [timm](https://github.com/rwightman/pytorch-image-models).
Thanks the authors for sharing their awesome projects.
