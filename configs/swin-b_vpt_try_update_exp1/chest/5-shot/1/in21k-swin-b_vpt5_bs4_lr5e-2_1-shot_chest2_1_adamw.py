_base_ = [
    '../../../../_base_/datasets/chest.py',
    '../../../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/custom_imports.py',
]


n = 1
vpl = 5
dataset = 'chest'
exp_num = 2
nshot = 5
lr = 5e-2
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
        type='MultiLabelLinearClsHead', num_classes=19, in_channels=1024))

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


log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = f'work_dirs/chest/{nshot}-shot/exp{exp_num-1}/{run_name}/best.pth'
work_dir = f'work_dirs/chest/{nshot}-shot/exp{exp_num}/{run_name}'


optimizer = dict(lr=lr)
runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
