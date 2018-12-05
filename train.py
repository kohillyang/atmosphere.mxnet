from mxnet.gluon.data.dataset import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.vision import transforms as transforms_mx
import os, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt
import mxnet.ndarray as nd
import gluoncv
import mxnet as mx
from mxnet import autograd as ag
from mxnet.metric import Accuracy
from mxnet import gluon
import time, logging
from gluoncv.utils import LRScheduler


class TestDataset(Dataset):
    def __init__(self, train_root, test_root, transforms=None):
        self.train_imgs = [x.strip().split('\t')[0] for x in open(os.path.join(train_root, "train.txt"), "rt")]
        self.test_imgs = [x.strip() for x in open(os.path.join(test_root, "image.txt"), "rt")]
        self.train_img_classes = {x.strip().split('\t')[0]: x.strip().split('\t')[1]
                                  for x in open(os.path.join(train_root, "train.txt"), "rt")}
        self.train_root = train_root
        self.test_root = test_root
        self.transforms = transforms

    def __getitem__(self, idx):
        if idx < len(self.test_imgs):
            path = self.test_imgs[idx]
            path = os.path.join(self.test_root, "test", path)
        else:
            path = self.train_imgs[idx - len(self.test_imgs)]
            path = os.path.join(self.train_root, "train", path)

        img = cv2.imread(path)[:, :, ::-1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, idx

    def get_image_names(self, idxes):
        all_images = []
        for idx in idxes:
            if idx < len(self.test_imgs):
                path = self.test_imgs[idx]
                all_images.append(["test", path])
            else:
                path = self.train_imgs[idx - len(self.test_imgs)]
                all_images.append([self.train_img_classes[path], path])

        return all_images

    def __len__(self):
        return len(self.train_imgs) + len(self.test_imgs)


def rotate_nobound(image, angle, center=None, scale=1.):
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def get_dataset():


    class to_ndarray(object):
        def __call__(self, x):
            return nd.array(x, dtype=np.float32)

    class to_numpy(object):
        def __call__(self, x):
            #             x = nd.transpose(x, axes = (2,0,1))
            return x.asnumpy()

    from data_aug import Compose, RandomHflip, ExpandBorder, RandomResizedCrop, Normalize, RandomRotate
    data_transforms = {
        'train': Compose([
            RandomHflip(),
            RandomRotate(angles=(-15, 15)),
            ExpandBorder(size=(368, 368), resize=True),
            RandomResizedCrop(size=(336, 336)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            ExpandBorder(size=(336, 336), resize=True),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    from dataset import ATDataset

    dataset_train = ATDataset(is_train=True).transform_first(data_transforms["train"])
    dataset_val = ATDataset(is_train=False).transform_first(data_transforms["val"])
    return dataset_train, dataset_val


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def get_net():
    from gluoncv.model_zoo import resnet50_v1b
    net = resnet50_v1b(pretrained=True, classes=1000, dilated=False, use_global_stats = True)
    net.fc = mx.gluon.nn.Dense(units=5)
    return net


def train(net, dataset_train, dataset_val, train_metric, metric_val, batch_size=16, log_interval=1000):
    ctx = mx.gpu(1)
    save_dir = "output"
    filehandler = logging.FileHandler(os.path.join(save_dir, "train_{}.log".format(time.time())))
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    # logger.info(dataset_train.transforms)
    # logger.info(dataset_val.transforms)

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8,  shuffle=True, last_batch="discard")
    val_loader = DataLoader(dataset_val, batch_size=16, num_workers=8, shuffle=True, last_batch="discard")
    lr_scheduler = LRScheduler("step", baselr=1e-3, niters=len(train_loader), nepochs=100,
                               step=[14, 24], step_factor=.1,
                               warmup_epochs=0)

    trainer = mx.gluon.Trainer(net.collect_params(),
                               'SGD',
                               {'learning_rate': 1e-3,
                                'multi_precision': True,
                                'lr_scheduler': lr_scheduler
                                },
                               )
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(100):
        tic = time.time()
        train_metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_loader):
            data, label = batch_fn(batch, ctx)

            with ag.record():
                outputs = [net(X.astype("float32", copy=False)) for X in data]
                loss = [L(yhat, y.astype("float32", copy=False)) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)

            train_metric.update(label, outputs)

            if i % 1000 == 0 and i > 0:
                train_metric_name, train_metric_score = train_metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                    epoch, i, batch_size * log_interval / (time.time() - btic),
                    train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        train_metric_name, train_metric_score = train_metric.get()
        throughput = int(batch_size * i / (time.time() - tic))

        # validate
        metric_val.reset()
        for batch in val_loader:
            data, label = batch_fn(batch, ctx)
            outputs = [net(X.astype("float32", copy=False)) for X in data]
            metric_val.update(label, outputs)
        val_accu = metric_val.get()[1]
        logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
        logger.info('[Epoch %d] validation: err-top1=%f ' % (epoch, val_accu))

        net.save_parameters('%s/imagenet-%s-%d-%.4f.params' % (save_dir, "resnet50", epoch, val_accu))


def inference(net: gluoncv.model_zoo.ResNetV1b, dataset, gpu_id=8):
    class Resnet(mx.gluon.nn.Block):
        def __init__(self, net):
            super(Resnet, self).__init__()
            net.load_parameters("/data1/zyx/yks/zero_short/classify/output/imagenet-resnet50-18-0.7463.params")
            net.collect_params().reset_ctx(mx.gpu(gpu_id))
            self.net = net

        def forward(self, x):
            feat = self.net
            x = feat.conv1(x)
            x = feat.bn1(x)
            x = feat.relu(x)
            x = feat.maxpool(x)

            x = feat.layer1(x)
            x = feat.layer2(x)
            x = feat.layer3(x)
            x = feat.layer4(x)

            x = feat.avgpool(x)
            x = feat.flat(x)
            if feat.drop is not None:
                x = feat.drop(x)
            return x

    net = Resnet(net)
    loader = DataLoader(dataset, batch_size=16, last_batch="keep", shuffle=False)
    features_all = []
    labels_all = []
    images_all = []
    from tqdm import tqdm
    for batch in tqdm(loader):
        data = batch[0].as_in_context(mx.gpu(gpu_id))
        idxes = batch[1].asnumpy().tolist()
        outputs = net(data).asnumpy()
        for feature, img_info in zip(outputs, dataset_test.get_image_names(idxes)):
            label, name = img_info
            features_all.append(feature[np.newaxis])
            labels_all.append(label)
            images_all.append(name)

    features_all = np.concatenate(features_all, axis=0)
    print(features_all.shape)
    data_all = {'features_all': features_all, 'labels_all': labels_all,
                'images_all': images_all}
    pickle.dump(data_all, open("feature_gluon_resnet50v1b_result.pkl", "wb"))


if __name__ == '__main__':
    net = get_net()
    metric_train = Accuracy(name="train_accu")
    metric_val = Accuracy(name="val_accu")
    dataset_train, dataset_val = get_dataset()
    # inference(net, dataset_test)
    train(net,dataset_train,dataset_val,metric_train,metric_val)
    #
    # da = pickle.load(open("/data1/zyx/yks/zero_short/classify/features.pickle", "rb"))
    # print(da["features_all"].shape)
    # print(da["labels_all"])
    # # print(da["images_all"])
    #
    # da = pickle.load(open("/data1/zyx/yks/zero_short/classify/feature_gluon_resnet50v1b_result.pkl", "rb"))
    # print(da["features_all"].shape)
    # print(da["labels_all"])
    # # print(da["images_all"])