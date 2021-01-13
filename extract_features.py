from __future__ import  absolute_import
import os
import torch as t
import numpy as np
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from tqdm import tqdm
from data.dataset import NewTestDataset
from torch.utils import data as data_


def demo(**kwargs):
    img = read_image('misc/demo.jpg')
    img = t.from_numpy(img)

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('pretrained/chainer_best_model_converted_to_pytorch_0.7053.pth')
    opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
    roi_topn, score_topn, feature_topn = trainer.faster_rcnn.get_roi_features(img, topN=50)
    print('Done')


def run(**kwargs):

    opt._parse(kwargs)
    # configurate result path
    result_path = os.path.join(opt.save_dir, opt.split)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # prepare dataset
    print('load data')
    testset = NewTestDataset(opt, split=opt.split)
    dataloader = data_.DataLoader(testset, batch_size=1, num_workers=opt.test_num_workers, shuffle=False, pin_memory=True)

    # prepare the model
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('pretrained/chainer_best_model_converted_to_pytorch_0.7053.pth')
    opt.caffe_pretrain=True # this model was trained from caffe-pretrained model

    for ii, (imgs, sizes, gt_bboxes, gt_labels, gt_difficults, names) in tqdm(enumerate(dataloader), total=len(dataloader), desc='%s split'%(opt.split)):
        # run inference for feature extraction
        rois, scores, features = trainer.faster_rcnn.get_roi_features(imgs[0], topN=50)
        # save the results
        imgID = '%06d'%(int(names[0]))
        result_file = os.path.join(result_path, imgID)
        np.savez(result_file, rois=rois, scores=scores, features=features, 
                 imgID=imgID, gtBoxes=gt_bboxes[0].numpy(), gtLabels=gt_labels[0].numpy(), gtDiff=gt_difficults[0].numpy())
    
    print('All results are saved in: %s'%(opt.save_dir))


if __name__ == "__main__":
    """
    python extract_features.py run --env fasterrcnn --split test --save_dir /ssd/VOC2007/VOC2007_Results 
    """
    import fire

    fire.Fire()