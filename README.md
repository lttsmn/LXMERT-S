# LXMERT-S:Multi-stage Pre-training over Simplified Multimodal Pre-training Models
## Introduction
Code for ACL/IJCNLP 2021 paper  ["LXMERT-S:Multi-stage Pre-training over Simplified Multimodal Pre-training Models"](https://arxiv.org/abs/2107.14596). 
## Fine-tuning
### VQA
1. Pre-training model parameter Download. You can download the Pre-training model from ["here"](https://pan.baidu.com/s/1HJd_n7vySqXTWscUSo1oxw?pwd=o025), Put mode_LXRT.bin under the data/ folder.
2. Data download. You can download the dataset required for VQA fine-tuning from ["LXMERT"](https://github.com/airsplay/lxmert#vqa).
3. Fine-tuning on VQA V2.0. You can also download our fine-tuned model parameters from ["here"](https://pan.baidu.com/s/1hk-4TFhzokBAs980WGgGgg?pwd=sp32).
   ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr050
   ```
4. Submitted to VQA test server.  You can also download our test results from ["here"](https://pan.baidu.com/s/1ME5ioNxTaGMlnQCijEddtg?pwd=ve5k)
    ```bash
    bash run/vqa_test.bash 0 vqa_lxr050_results --test test --load snap/vqa/vqa_lxr050/BEST
    ```
5. Results: 
   
    | results | Local Validation | Test-Dev | Test-Standard | 
    |-----------       |:----:   |:---:    |:------:|
    | [VQA](https://visualqa.org/)| 68.49%  | 71.1%  | 71.18% |
### GQA
1. Pre-training model parameter Download. You can download the Pre-training model from [here](https://pan.baidu.com/s/1HJd_n7vySqXTWscUSo1oxw?pwd=o025), Put mode_LXRT.bin under the data/ folder.
2. Data download. You can download the dataset required for VQA fine-tuning from [LXMERT](https://github.com/airsplay/lxmert#gqa).
3. Fine-tuning on GQA. You can also download our fine-tuned model parameters from [here](https://pan.baidu.com/s/1HPnD2GC3konAQEzv0sTOiA?pwd=c9vm).
   ```bash
    bash run/gqa_finetune.bash 0 gqa_lxr050
   ```
4. Submitted to GQA test server.  You can also download our test results from [here]()
    ```bash
    bash run/gqa_test.bash 0 vqa_lxr050_results --test test --load snap/gqa/gqa_lxr050/BEST
    ```
5. Results: 
    | results | Test-Dev | Test-Standard | 
    |-----------    |:---:    |:------:|
    | [GQA]()| 58.7%  | 59.12% |
### NLVR2
### IR
1.  Pre-training model parameter Download. You can download the Pre-training model from [here](https://pan.baidu.com/s/1HJd_n7vySqXTWscUSo1oxw?pwd=o025), Put mode_LXRT.bin under the data/ folder.
2.  Zero shot.
    ```bash
    bash run/image_retrieval_zero_shot.bash 0 ir_zs
    ```
3. Fine-tuning on IR.  You can execute the following instructions or download our fine-tuned model parameters from [here](https://pan.baidu.com/s/1KIB2nX5o6ObB6QE1WVkvpw?pwd=q43p).
   ```bash
    bash run/image_retrieval_finetune.bash 0 ir_lxr050
   ```
4. Results: 
    | results | R@1 | R@5 | R@10 | 
    |-----------       |:----:   |:---:    |:------:|
    | ir(zero_shot)| 42.42%  | 68.7%  | 77.92% |
    | ir(finetune)| 57.9%  | 83%  | 88.7% |
### TR
1.  Pre-training model parameter Download. You can download the Pre-training model from [here](https://pan.baidu.com/s/1HJd_n7vySqXTWscUSo1oxw?pwd=o025), Put mode_LXRT.bin under the data/ folder.
2. Zero shot.
    ```bash
    bash run/sent_retrieval_zero_shot.bash 0 tr_zs
    ```
3. Fine-tuning on IR.  You can execute the following instructions or download our fine-tuned model parameters from [here](https://pan.baidu.com/s/1KIB2nX5o6ObB6QE1WVkvpw?pwd=q43p).
   ```bash
    bash run/sent_retrieval_finetune.bash 0 ir_lxr050
   ```
4. Results:     
    | results | R@1 | R@5 | R@10 | 
    |-----------       |:----:   |:---:    |:------:|
    | tr(zero_shot)| 49%  | 75%  | 81.8% |
    | tr(finetune)| 64.6%  | 87.5%  | 90.4% |

## Reference
If you find this project helps, please cite our paper :)

```bibtex
@inproceedings{liu2021acl,
  author    = {Tongtong Liu and
               Fangxiang Feng and
               Xiaojie Wang},
  title     = {Multi-stage Pre-training over Simplified Multimodal Pre-training Models},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
               Linguistics and the 11th International Joint Conference on Natural
               Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
               Event, August 1-6, 2021},
  year      = {2021},
}
```
## Acknowledgement
We thank [airsplay](https://github.com/lovelyzzc/lxmert) for providing the LXMERT code and pre-trained models. We thank [jiasenlu](https://github.com/jiasenlu/vilbert_beta) for providing the ViLBERT code.
   
   
   
   
   
   
   
