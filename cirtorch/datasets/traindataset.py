import os
import pickle
import pdb

import torch
import torch.utils.data as data

from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of 
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    参数列表
    Args:
        指定数据集名
        name (string): dataset name: 'retrieval-sfm-120k'
        数据集类型 train还是val
        mode (string): 'train' or 'val' for training and validation parts of dataset
        图像缩放后的大小，最长边
        imsize (int, Default: None): Defines the maximum size of longer image side
        图像转换
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        数据加载函数，默认为default_loader (PIL方式加载RGB格式数据)
        loader (callable, optional): A function to load an image given its path.
        负样本个数，默认为5个
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        查询图像数目，也就是每一轮参与训练的图像对数目，默认为2000
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        进行负样本挖掘的图像列表数目，每一轮重新生成，默认为20000
        poolsize (int, Default:10000): Pool size for negative images re-mining

     属性列表
     Attributes:
        每个图像的完整路径列表
        images (list): List of full filenames for each image
        每个图像对应的聚类ID，因为采用了无监督方式进行数据聚类，所以聚类ID等同于标签
        clusters (list): List of clusterID per image
        所有查询数据的下标列表
        qpool (list): List of all query image indexes
        每个查询数据对应的正样本数据的下标列表
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        每一轮中参与训练的查询数据下标列表，每一轮都重新生成
        qidxs (list): List of qsize query image indexes to be processed in an epoch
        每一轮查询数据对应的正样本数据的下标列表
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        每一轮查询数据对应的负样本数据的下标列表
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        注意：每一轮训练开始之前需要调用函数create_epoch_tuples()重新生成该轮参与训练的查询数据、正样本和负样本
        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = os.path.join(db_root, 'ims')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            ## TODO: NOT IMPLEMENTED YET PROPOERLY (WITH AUTOMATIC DOWNLOAD)

            # setting up paths
            db_root = '/mnt/fry2/users/datasets/landmarkscvprw18/recognition/'
            ims_root = os.path.join(db_root, 'images', 'train')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [os.path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]
        else:
            raise(RuntimeError("Unknown dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        # 该数据集指定的查询列表
        self.qpool = db['qidxs']
        # 该数据集指定的正样本数据列表
        # 注意：从下面结果来看，正样本和查询数据是一一对应的。也就是说
        # len(self.qpool) == len(self.pool)
        self.ppool = db['pidxs']

        ## If we want to keep only unique q-p pairs 
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.nnum = nnum
        # 指定每一轮参与训练的查询数据数目
        self.qsize = min(qsize, len(self.qpool))
        # 指定每一轮参与苦难负样本挖掘的数据池数目
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        # 加载查询数据
        output.append(self.loader(self.images[self.qidxs[index]]))
        # positive image
        # 加载正样本
        output.append(self.loader(self.images[self.pidxs[index]]))
        # negative images
        # 加载负样本
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        # 图像缩放
        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        # 图像转换
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        # 查询数据标签为-1
        # 正样本标签为1
        # 负样本标签为0
        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

        return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, net):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
        print(">>>> used network: ")
        print(net.meta_repr())

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        # 随机打乱查询数据集，选取前qsize张作为本轮训练的数据
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        # 指定查询数据下标
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        # 指定正样本数据下标
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # 选择poolsize张图像作为负样本挖掘的数据集
        # draw poolsize random images for pool of negatives images
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            # prepare query loader
            # 创建查询数据加载器
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            # 提取查询特征向量
            # extract query vectors
            # 提取查询数据的特征向量，大小为[D, N_q]
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                qvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            print('')

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            # 创建负样本数据集加载器
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract negative pool vectors
            # 提取负样本数据集的特征向量，大小为[D, N_pool]
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2images):
                    print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            print('')

            print('>> Searching for hard negatives...')
            # 负样本挖掘
            # 计算查询数据和负样本的相似度 [N_pool, D] x [D, N_q] = [N_pool, N_q]
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            # 按照相似度从大到小进行排序
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            # 针对每个查询图像，挖掘负样本
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                # 找出查询图像对应的聚类ID，也就是对应标签
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                # 采集nnum张负样本
                while len(nidxs) < self.nnum:
                    # 按照相似度从大到小方式进行遍历
                    potential = idxs2images[ranks[r, q]]
                    # 如果候选数据对应的聚类ID等于查询数据或者已加入的负样本标签，则跳过
                    # take at most one image from the same cluster
                    if not self.clusters[potential] in clusters:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        # 计算该负样本特征与查询数据特征的欧式距离
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')

        return (avg_ndist/n_ndist).item()  # return average negative l2-distance
