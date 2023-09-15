import mmcv
import numpy as np
import os
import torch
from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from tqdm import tqdm

def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
    return scores


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_file',default='data/MedFMC_test/', help='Names of test image files')
    parser.add_argument('--img_path', default='data/MedFMC_test/',help='Path of test image files')
    parser.add_argument('--config',default=r'../..\configs\swin-b_vpt_try\chest\10-shot\in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_2_adamw.py', help='Config file')
    parser.add_argument('--checkpoint',default=r'../tools\work_dirs\chest\10-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best_AUC_multilabel_epoch_19.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--output-prediction',
        help='where to save prediction in csv file',
        default='chest_10-shot_submission.csv')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    chest_config_list = [r"..\configs\swin-b_vpt_try_update_exp1\chest\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\chest\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest5_2_adamw.py",
                   ]

    colon_config_list = [r"..\configs\swin-b_vpt_try_update_exp1\colon\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\colon\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_colon5_2_adamw.py",
                   ]

    endo_config_list = [r"..\configs\swin-b_vpt_try_update_exp1\endo\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\1-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\5-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo5_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo1_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo2_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo3_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo4_2_adamw.py",
                   r"..\configs\swin-b_vpt_try_update_exp1\endo\10-shot\2\in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_endo5_2_adamw.py",
                   ]


    chest_checkpoint_list = [r"tools\work_dirs\chest\1-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_chest\best.pth",
                       r"tools\work_dirs\chest\1-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_chest\best.pth",
                       r"tools\work_dirs\chest\1-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_chest\best.pth",
                       r"tools\work_dirs\chest\1-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_chest\best.pth",
                       r"tools\work_dirs\chest\1-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_chest\best.pth",
                       r"tools\work_dirs\chest\5-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_chest\best.pth",
                       r"tools\work_dirs\chest\5-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_chest\best.pth",
                       r"tools\work_dirs\chest\5-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_chest\best.pth",
                       r"tools\work_dirs\chest\5-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_chest\best.pth",
                       r"tools\work_dirs\chest\5-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_chest\best.pth",
                       r"tools\work_dirs\chest\10-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best.pth",
                       r"tools\work_dirs\chest\10-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best.pth",
                       r"tools\work_dirs\chest\10-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best.pth",
                       r"tools\work_dirs\chest\10-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best.pth",
                       r"tools\work_dirs\chest\10-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest\best.pth",
                       ]
    colon_checkpoint_list = [r"tools\work_dirs\colon\1-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_colon\best.pth",
                       r"tools\work_dirs\colon\1-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_colon\best.pth",
                       r"tools\work_dirs\colon\1-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_colon\best.pth",
                       r"tools\work_dirs\colon\1-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_colon\best.pth",
                       r"tools\work_dirs\colon\1-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_colon\best.pth",
                       r"tools\work_dirs\colon\5-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_colon\best.pth",
                       r"tools\work_dirs\colon\5-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_colon\best.pth",
                       r"tools\work_dirs\colon\5-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_colon\best.pth",
                       r"tools\work_dirs\colon\5-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_colon\best.pth",
                       r"tools\work_dirs\colon\5-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_colon\best.pth",
                       r"tools\work_dirs\colon\10-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_colon\best.pth",
                       r"tools\work_dirs\colon\10-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_colon\best.pth",
                       r"tools\work_dirs\colon\10-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_colon\best.pth",
                       r"tools\work_dirs\colon\10-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_colon\best.pth",
                       r"tools\work_dirs\colon\10-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_colon\best.pth",
                       ]
    endo_checkpoint_list = [r"tools\work_dirs\endo\1-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_endo\best.pth",
                       r"tools\work_dirs\endo\1-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_endo\best.pth",
                       r"tools\work_dirs\endo\1-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_endo\best.pth",
                       r"tools\work_dirs\endo\1-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_endo\best.pth",
                       r"tools\work_dirs\endo\1-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_1-shot_endo\best.pth",
                       r"tools\work_dirs\endo\5-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_endo\best.pth",
                       r"tools\work_dirs\endo\5-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_endo\best.pth",
                       r"tools\work_dirs\endo\5-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_endo\best.pth",
                       r"tools\work_dirs\endo\5-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_endo\best.pth",
                       r"tools\work_dirs\endo\5-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_5-shot_endo\best.pth",
                       r"tools\work_dirs\endo\10-shot\exp1\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_endo\best.pth",
                       r"tools\work_dirs\endo\10-shot\exp2\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_endo\best.pth",
                       r"tools\work_dirs\endo\10-shot\exp3\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_endo\best.pth",
                       r"tools\work_dirs\endo\10-shot\exp4\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_endo\best.pth",
                       r"tools\work_dirs\endo\10-shot\exp5\in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_endo\best.pth",
                       ]

    chest_name = ["exp1/chest_1-shot_submission.csv", "exp2/chest_1-shot_submission.csv",
                  "exp3/chest_1-shot_submission.csv", "exp4/chest_1-shot_submission.csv",
                  "exp5/chest_1-shot_submission.csv",
                  "exp1/chest_5-shot_submission.csv", "exp2/chest_5-shot_submission.csv",
                  "exp3/chest_5-shot_submission.csv", "exp4/chest_5-shot_submission.csv",
                  "exp5/chest_5-shot_submission.csv",
                  "chest_10-shot_submission.csv", "chest_10-shot_submission.csv", "exp3/chest_10-shot_submission.csv",
                  "exp4/chest_10-shot_submission.csv", "exp5/chest_10-shot_submission.csv"]

    colon_name = ["exp1/colon_1-shot_submission.csv", "exp2/colon_1-shot_submission.csv",
                  "exp3/colon_1-shot_submission.csv", "exp4/colon_1-shot_submission.csv",
                  "exp5/colon_1-shot_submission.csv",
                  "exp1/colon_5-shot_submission.csv", "exp2/colon_5-shot_submission.csv",
                  "exp3/colon_5-shot_submission.csv", "exp4/colon_5-shot_submission.csv",
                  "exp5/colon_5-shot_submission.csv",
                  "exp1/colon_10-shot_submission.csv", "exp2/colon_10-shot_submission.csv",
                  "exp3/colon_10-shot_submission.csv", "exp4/colon_10-shot_submission.csv",
                  "exp5/colon_10-shot_submission.csv"]

    endo_name = ["exp1/endo_1-shot_submission.csv", "exp2/endo_1-shot_submission.csv",
                 "exp3/endo_1-shot_submission.csv", "exp4/endo_1-shot_submission.csv",
                 "exp5/endo_1-shot_submission.csv",
                 "exp1/endo_5-shot_submission.csv", "exp2/endo_5-shot_submission.csv",
                 "exp3/endo_5-shot_submission.csv", "exp4/endo_5-shot_submission.csv",
                 "exp5/endo_5-shot_submission.csv",
                 "exp1/endo_10-shot_submission.csv", "exp2/endo_10-shot_submission.csv",
                 "exp3/endo_10-shot_submission.csv", "exp4/endo_10-shot_submission.csv",
                 "exp5/endo_10-shot_submission.csv"]

    cls_list = ["chest"]

    for cls in range(len(cls_list)):
        if cls_list[cls] == "chest":
            for i in range(len(chest_name)):
                config = chest_config_list[i]
                checkpoint = chest_checkpoint_list[i]
                name_1 = chest_name[i]
                model = init_model(config, checkpoint, device=args.device)
                # test a bundle of images
                with open(name_1, 'w') as f_out:
                    for line in tqdm(open(args.img_file+cls_list[cls]+"/test_WithoutLabel.txt", 'r')):
                        image_name = line.split('\n')[0]
                        file = os.path.join(args.img_path+cls_list[cls]+"/images", image_name)
                        result = inference_model(model, file)[0]
                        f_out.write(image_name)
                        for j in range(len(result)):
                            f_out.write(',' + str(np.around(result[j], 8)))
                        f_out.write('\n')

        elif cls_list[cls] == "colon":
            for i in range(len(colon_name)):
                config = colon_config_list[i]
                checkpoint = colon_checkpoint_list[i]
                name_1 = colon_name[i]
                model = init_model(config, checkpoint, device=args.device)
                # test a bundle of images
                with open(name_1, 'w') as f_out:
                    for line in tqdm(open(args.img_file+cls_list[cls]+"/test_WithoutLabel.txt", 'r')):
                        image_name = line.split('\n')[0]
                        file = os.path.join(args.img_path+cls_list[cls]+"/images", image_name)
                        result = inference_model(model, file)[0]
                        f_out.write(image_name)
                        for j in range(len(result)):
                            f_out.write(',' + str(np.around(result[j], 8)))
                        f_out.write('\n')
        else:
            for i in range(len(endo_name)):
                config = endo_config_list[i]
                checkpoint = endo_checkpoint_list[i]
                name_1 = endo_name[i]
                model = init_model(config, checkpoint, device=args.device)
                # test a bundle of images
                with open(name_1, 'w') as f_out:
                    for line in tqdm(open(args.img_file+cls_list[cls]+"/test_WithoutLabel.txt", 'r')):
                        image_name = line.split('\n')[0]
                        file = os.path.join(args.img_path+cls_list[cls]+"/images", image_name)
                        result = inference_model(model, file)[0]
                        f_out.write(image_name)
                        for j in range(len(result)):
                            f_out.write(',' + str(np.around(result[j], 8)))
                        f_out.write('\n')


if __name__ == '__main__':
    main()
