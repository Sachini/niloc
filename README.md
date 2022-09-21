# Neural Inertial Localization

**Paper**: [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Herath_Neural_Inertial_Localization_CVPR_2022_paper.html), [arXiv](https://arxiv.org/abs/2203.15851)   
**Website**: https://sachini.github.io/niloc  
**Demo**: https://youtu.be/FmkfUKhKe2Q

---
This is the implementation of the approach described in the paper.

>Herath, S., Caruso, D., Liu, C., Chen, Y. and Furukawa, Y., [Neural Inertial Localization](https://openaccess.thecvf.com/content/CVPR2022/html/Herath_Neural_Inertial_Localization_CVPR_2022_paper.html). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022.

We provide the code for reproducing our results, datasets as well as pre-trained models.  
- Dataset : [Dropbox](https://www.dropbox.com/scl/fo/uux0twqk7gsgwdpljkahd/h?dl=0&rlkey=0g8qi66jsl14ffbx6r7nfn3rx)  
- Models : _coming soon_

Please cite the following paper is you use the code, paper, models or data.

---
### Instructions 
 1. Setup conda environment from `niloc_env.yaml`
 2. Follow instructions on `preprocess/README.md` to preprocess real data and optionally, generate synthetic data.
 3. Setup necessary file paths in `niloc/config`. (dataset: dataset paths, grid: map image paths, io: output paths)
 4. [Optional] Pretrain using IMU + synthetic data. Parameters used in paper are set as defaults.
    ```
    ./train_synthetic.sh <building>
    ```
 5. Train using IMU data. [Optional] load pretrained weights.
    ```
    ./train_imu.sh <building> [<path to pretrained checkpoint>]
    ```
 6. Evaluate
    * Select the checkpoints to use and create checkpoint file described in `niloc/cmd_test_file.py`
    
    ```
    ./test_imu.sh <building> <checkpoint file>
    ```

Please refer to the code for advance configurations.