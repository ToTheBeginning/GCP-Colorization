## Model Description

| Models                 | Description          |  note |
| ---------------------- | :------------------ | :-----|
| biggan_D_256.pth       | BigGAN discriminator, for 256x256 images   | original from [DGP](https://github.com/XingangPan/deep-generative-prior)
| biggan_G_ema_256.pth   | BigGAN generator, for 256x256 images   | original from [DGP](https://github.com/XingangPan/deep-generative-prior)
| biggan_E.pth (default) | BigGAN encoder trained on the 1000 classes of ImageNet |
| colorization_model.pth (default) | colorization model trained on the 1000 classes of ImageNet | we got this model after the paper deadline. This model can get the best quantitative and qualitative results. If you want to test images outside the ImageNet val set, please use this model.
| biggan_E_100class.pth  | BigGAN encoder trained on first 100 classes of ImageNet (in lexicographical order) | use this [model](https://drive.google.com/drive/folders/1xf90Hq0Ce-OyFt6gVG9TKNsrKt_xfyPK?usp=sharing) only if you want yo reproduce some of the images in the paper
| colorization_model_100class.pth | colorization model trained on first 100 classes of ImageNet  | we use this model and biggan_E.pth to calculate the quantitative metrics in the paper. use this [model](https://drive.google.com/drive/folders/1xf90Hq0Ce-OyFt6gVG9TKNsrKt_xfyPK?usp=sharing) only if you want yo reproduce some of the images in the paper
| colorization_model_selectclass.pth | colorization model finetuned on selected class, see README in [[Tencent Cloud 腾讯微云](), code: 2i4idb]] for details. | use this [model](https://drive.google.com/drive/folders/1xf90Hq0Ce-OyFt6gVG9TKNsrKt_xfyPK?usp=sharing) only if you want yo reproduce some of the images in the paper |


## Some Notes
* If you want to make visual comparison with our method, or the test images are not in ImageNet val set, please use the default model, i.e., `biggan_E.pth` and `colorization_model.pth`.
