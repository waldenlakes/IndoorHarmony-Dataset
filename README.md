# IndoorHarmony-Dataset
The IndoorHarmony-Dataset for Spatially-Varying Illumination-Aware Indoor Harmonization (IJCV2024).

## Dataset Overview
We constructed a large-scale synthetic image harmonization dataset (called **IndoorHarmony-Dataset** by us) where the foreground focuses on humans and is perturbed and rendered by spatially varying illuminations. 

## Downloading Data
The IndoorHarmony-Dataset (\~135GB) can be downloaded from [OneDrive](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/zy_h_mail_nwpu_edu_cn/EmFXyW07gqxHvGWdOUz5WREBH6VJrHpYSybL5wDrTpJVmw?e=OSKGbR). This dataset is licensed under a [LICENSE](./LICENSE).

## Loading Data
We provide python scripts in the foloder [./data_utils](./data_utils) for loading data.

***loading training data.*** By running this script, you can get the composite training data from the rendered raw data. Note that for the input unharmonized composite image, its foreground illumination is randomly selected to increase the diversity of the IndoorHarmony-Dataset.

    python data_utils/train_dataloader.py

***loading test data.*** Run the following script to get the input unharmonized image, mask and ground-truth.

    python data_utils/test_dataloader.py

We also provided [our test results](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/zy_h_mail_nwpu_edu_cn/ERkOg_XmLEtAiU247igPF9IBhJz3Lal-7LNYaw4G3fGp1Q?e=bGLFgO) on the test set to facilitate possible future comparative studies.

## Real Data from User Study
In addition to the IndoorHarmony-Dataset, we also built [a small indoor harmonization dataset](https://drive.google.com/file/d/1EyHf6KdT2A4De1eAUr-tPAxf6M2iLxlV/view?usp=sharing) consisting of 52 real composite images for user study. We also provided [our results](https://drive.google.com/file/d/19WKXW1GoUKwXqgxxM05roHfE1PbKk_jh/view?usp=sharing) on this small dataset.

## Citation

If you use these data in your research, please cite:

```
@article{hu2024sv,
  title={Spatially-Varying Illumination-Aware Indoor Harmonization},
  author={Hu, Zhongyun and Li, Jiahao and Wang, Xue and Wang, Qing},
  journal={International Journal of Computer Vision},
  year={2024}
}
```

## Contact
If you have any questions, please contact <zy_h@mail.nwpu.edu.cn>.
