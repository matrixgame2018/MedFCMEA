_base_ = [
    '../../../../_base_/datasets/colon.py',
    '../../../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/custom_imports.py',
]

lr = 5e-2
n = 1
vpl = 5
dataset = 'colon'
exp_num = 1
nshot = 1
run_name = f'in21k-swin-b_vpt-{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer_base',
        prompt_length=vpl,
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ))

data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        data_prefix=
        f'data/MedFMC_train/{dataset}/images',
        ann_file=
        f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'
    ),
    val=dict(
        data_prefix=
        f'data/MedFMC_train/{dataset}/images',
        ann_file=
        f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
    test=dict(
        data_prefix=f'data/MedFMC_train/{dataset}/images',
        ann_file=f'data/MedFMC_train/{dataset}/test_WithLabel.txt'))



optimizer = dict(lr=lr)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = 'pretrain/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'
work_dir = f'work_dirs/colon/{nshot}-shot/exp{exp_num}/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
