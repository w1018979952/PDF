import configargparse
import data_loader
import os
import torch
import torch.nn as nn
import models
import utils
import sys
from utils import str2bool
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import torch.utils.data as data
from domain_losses import ProtoLoss
from lr_scheduler import StepwiseLR

eps = sys.float_info.epsilon

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    #parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--config",type=str, default='code/DeepDA/DANN/DANN.yaml')
    parser.add("--seed", type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=3)
    
    # network related
    # parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--backbone', type=str, default='swimvit')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    # parser.add_argument('--data_dir', type=str, default='/data/guoshuo/data/')
    parser.add_argument('--data_dir', type=str, default='/DATA/hejinbo/qiyuhan/datasets/')
    parser.add_argument('--dataset', type=str, default='Oulu')
    parser.add_argument('--src_domain', type=str, default='visible')
    parser.add_argument('--tgt_domain', type=str, default='infrared')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_epoch', type=int, default=70)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=300, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)
    
    # domain losses
    parser.add_argument('-nav_t', '--nav_t', default=1, type=float,
                        help='temperature for the navigator')
    parser.add_argument('-beta', '--beta', default=0, type=float,
                        help='momentum coefficient')
    parser.add_argument('--s_par', default=0.5, type=float, 
                        help='s_par')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=1.0)
    parser.add_argument('--transfer_loss', type=str, default='adv')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataSet_Choice(data.Dataset):
    def __init__(self, dataset, transform = None,mode ='train'):
        # Load training images and labels
        
        self.transform = transform
        self.raf_path = ''
        if dataset =='NVIE':
            self.raf_path = '/DATA/hejinbo/qiyuhan/USTC_NVIE/'
        else:
            self.raf_path = '/DATA/hejinbo/qiyuhan/datasets/'
        if dataset == 'NVIE':
            if mode == 'source_train':
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/USTC_NVIE/SpontaneousDatabase/source.txt")
            elif mode == 'target_train':
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/USTC_NVIE/SpontaneousDatabase/target_train.txt")
            else:
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/USTC_NVIE/SpontaneousDatabase/target_test.txt")
        else:
            if mode == 'source_train':
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/datasets/Oulu_CASIA_NIR_VIS/VL/oulu_CASIA_all_train.txt")
            elif mode == 'target_train':
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/datasets/Oulu_CASIA_NIR_VIS/VL/oulu_CASIA_dark_RGB_train.txt")
            else:
                file_path = os.path.join("/DATA/hejinbo/qiyuhan/datasets/Oulu_CASIA_NIR_VIS/VL/oulu_CASIA_dark_RGB_test.txt")

        self.train_img = []
        self.train_id = []
        with open(file_path, 'r') as file:
            line = file.readlines()
            for row in line:    
                tmp_list = row.split(' ')
                self.train_img.append(self.raf_path + tmp_list[0])
                self.train_id.append(int(tmp_list[1][0]))
                            
    def __len__(self):
        return len(self.train_img)   
                         
    def __getitem__(self, index):
        path = self.train_img[index]
        image = Image.open(path)
        # image1 = Image.open(path).convert('RGB')
        label = self.train_id[index]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck,device1=args.device).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    #optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def calculate_entropy(histogram):  
    """  
    计算直方图的信息熵  
    """  
    # 将直方图归一化为概率分布  
    probs = histogram / (np.sum(histogram)+0.01) 
    # 避免log(0)的情况  
    probs = probs + np.finfo(float).eps  
    entropy = -np.sum(probs * np.log2(probs))  

    return entropy

def mixup_loss(mixed_x, src_labels, tgt_pseudo, lam):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    return lam * criterion(mixed_x, src_labels) + (1 - lam) * criterion(mixed_x, tgt_pseudo)

class PartitionLoss(torch.nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss

def test1(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)
    CELoss = torch.nn.CrossEntropyLoss()
    criterion_pt = PartitionLoss()
    
    domain_loss = ProtoLoss(args.nav_t, args.beta, 6, args.device, args.s_par).to(args.device)
    beta_scheduler = StepwiseLR(None, init_lr=args.beta, gamma=args.lr_gamma, decay_rate=0.75)
    
    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch+1):
        model.train()
        domain_loss.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_domain = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        train_loss_mixup = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        for _ in range(n_batch):
            
            beta_scheduler.step()
            
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            
            source, source_scord, target, target_scord, transfer_loss, heads_s= model(data_source, data_target)
            
            prototypes_s = model.classifier_layer.weight.data.clone()
            
            #source_scord = model.predict(data_source)
            clf_loss1 = CELoss(source_scord, label_source)
            
            domain_loss.beta = beta_scheduler.get_lr()
            domain_loss1 = domain_loss(prototypes_s, target)
            
            atte_loss = criterion_pt(heads_s)

            #BNM
            #softmax_tgt = nn.Softmax(dim=1)(target_scord)
            #_, s_tgt, _ = torch.svd(softmax_tgt)
            #bnm_loss = -torch.mean(s_tgt)
            
             #剪枝mixup
            ent_sourcelist=[]
            ent_targetlist=[]

            idx=0
            for img in data_source:
                img_array= np.array(img.cpu())
                entropies = []  
                for channel in range(3):  # RGB三个通道  
                    a = img_array[:, :, channel].min() 
                    b =  img_array[:, :, channel].max()   
                    hist, _ = np.histogram(img_array[:, :,channel].flatten(), bins=100, range=(a, b))  
                    # 计算信息熵  
                    entropy = calculate_entropy(hist)  
                    entropies.append(entropy) 
                score = sum(entropies)
                ent_sourcelist.append((idx,score))
                idx=idx+1

            idx=0
            for img in data_target:
                img_array= np.array(img.cpu())
                entropies = []  
                for channel in range(3):  # RGB三个通道  
                    a = img_array[:, :, channel].min() 
                    b =  img_array[:, :, channel].max() 
                    hist, _ = np.histogram(img_array[:, :,channel].flatten(), bins=100, range=(a,b))  
                    # 计算信息熵  
                    entropy = calculate_entropy(hist)  
                    entropies.append(entropy) 
                score = sum(entropies)
                ent_targetlist.append((idx,score))
                idx=idx+1
            sorted_sourcelist = sorted(ent_sourcelist, key=lambda x: x[1], reverse=True)[:15]
            sorted_targetlist = sorted(ent_targetlist, key=lambda x: x[1], reverse=True)[:15]
            num_source = [item[0] for item in sorted_sourcelist]    #信息量前10的source
            num_target = [item[0] for item in sorted_targetlist]    #信息量前10的target
            top_source = []
            top_slabel=[]
            top_target = []
            top_tlabel = []
            pred = torch.max(target_scord, 1)[1]
            for id_source in num_source:
                top_source.append(data_source[id_source])
                top_slabel.append(label_source[id_source])
            for id_target in num_target:
                top_target.append(data_target[id_target]) 
                top_tlabel.append(pred[id_target])  
            
            ratio = 0.7
            top_source1 = torch.stack(top_source, dim=0)  
            top_target1 = torch.stack(top_target, dim=0) 
            top_slabel = torch.stack(top_slabel, dim=0)
            top_tlabel = torch.stack(top_tlabel, dim=0)
            mixed_x = ratio * top_source1 + (1-ratio) * top_target1
            _,mixed_x,_,_,_,_=model(mixed_x, mixed_x)
            # _,_,_,top_tscore,_,_=model(mixed_x, mixed_x)
            
            mix_loss = mixup_loss(mixed_x, top_slabel, top_tlabel, ratio)
            
            loss = clf_loss1 + args.transfer_loss_weight * transfer_loss + atte_loss + domain_loss1 + mix_loss
            # loss = clf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss1.item())
            train_loss_domain.update(domain_loss1.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_mixup.update(mix_loss.item())
            train_loss_total.update(loss.item())
            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, train_loss_domain.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, prototype_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_domain.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test1(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: {:.4f}'.format(best_acc))

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    print(args)
    set_random_seed(args.seed)
    
    train_transforms = transforms.Compose([
            transforms.Resize(256),#[256, 256]S
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    train_dataset = DataSet_Choice( args.dataset , transform = train_transforms,mode ='source_train')
    source_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = True,  
                                               pin_memory = True,
                                               drop_last = True) 
    target_train_dataset = DataSet_Choice( args.dataset , transform = train_transforms,mode ='target_train')
    print('Whole train set size:', train_dataset.__len__())
    
    target_train_loader= torch.utils.data.DataLoader(target_train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = True,  
                                               pin_memory = True,
                                               drop_last = True)
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    target_test_dataset = DataSet_Choice( args.dataset , transform = train_transforms,mode ='target_test')
    target_test_loader = torch.utils.data.DataLoader(target_test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)
    print('Validation set size:', target_test_dataset.__len__())
    #source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", 6)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    model.to(args.device)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
