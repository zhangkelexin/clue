import os
import torch
import argparse
import random
import pandas as pd
import numpy as np
import pdb
import tqdm
import torch.nn.functional as F


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # 生成器的配置
    parser.add_argument('--gen_drop', default=0.5, type=float)
    parser.add_argument('--gen_std', default=1, type=float)

    # Tunable
    parser.add_argument('--model', default='RotatE', type=str)  # 模型
    parser.add_argument('--num_ng', default=64, type=int)  # 负样本个数
    parser.add_argument('--bs', default=512, type=int)  # batch_size
    parser.add_argument('--emb_dim', default=1024, type=int)  # 维度
    parser.add_argument('--lrd', default=0.0001, type=float)  # 学习率
    parser.add_argument('--reg', default=0.001, type=float)  # 正则
    parser.add_argument('--refer', default=0.5, type=float)  # 参考答案比例
    parser.add_argument('--refer_num', default=4, type=int)  # 参考答案个数

    # Misc
    open_set = 2
    finally_set = './data/FB15k-237' if open_set == 1 else './data/WN18RR'
    parser.add_argument('--data_root', default=finally_set, type=str)  # 数据集
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--seed', default=42, type=int)  # 随机数种子
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)  # 一共训练多少轮
    parser.add_argument('--valid_interval', default=2, type=int)  # 每隔几轮测试一次
    return parser.parse_args(args)


def read_data(root):
    # read entity dict and relation dict
    e_dict = {}
    r_dict = {}
    e_data = pd.read_csv(root + 'entities.dict', header=None, delimiter='\t').values
    r_data = pd.read_csv(root + 'relations.dict', header=None, delimiter='\t').values
    for record in r_data:
        r_dict[record[1]] = record[0]
    for record in e_data:
        e_dict[record[1]] = record[0]

    # read data and map to index
    train_data = pd.read_csv(root + 'train.txt', header=None, delimiter='\t')
    valid_data = pd.read_csv(root + 'valid.txt', header=None, delimiter='\t')
    test_data = pd.read_csv(root + 'test.txt', header=None, delimiter='\t')
    for data in [train_data, valid_data, test_data]:
        for column in range(3):
            if column != 1:
                data[column] = data[column].map(e_dict)
            else:
                data[column] = data[column].map(r_dict)
        data.columns = ['h', 'r', 't']

    # already existing heads or tails (for sampling and evaluation)
    already_ts_dict = {}
    already_hs_dict = {}
    already_ts = train_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs = train_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts:
        already_ts_dict[(record[0], record[1])] = record[2]
    for record in already_hs:
        already_hs_dict[(record[0], record[1])] = record[2]

    all_data = pd.concat([train_data, valid_data, test_data])
    already_ts_dict_all = {}
    already_hs_dict_all = {}
    already_ts_all = all_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs_all = all_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts_all:
        already_ts_dict_all[(record[0], record[1])] = record[2]
    for record in already_hs_all:
        already_hs_dict_all[(record[0], record[1])] = record[2]

    return e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict, already_ts_dict_all, already_hs_dict_all


def get_gen_neg(h_emb, r_emb, t_emb, gen, bs, num_ng, emb_dim, device, gen_std, flag):
    z_tail = torch.normal(mean=0, std=gen_std, size=(bs, num_ng // 2, emb_dim // 8)).to(device)
    z_head = torch.normal(mean=0, std=gen_std, size=(bs, num_ng // 2, emb_dim // 8)).to(device)
    if flag == 'gen':
        neg_gen_tail = gen(z_tail)
        neg_gen_head = gen(z_head)
        h_emb, r_emb, t_emb = h_emb.detach(), r_emb.detach(), t_emb.detach()
    elif flag == 'dis':
        neg_gen_tail = gen(z_tail).detach()
        neg_gen_head = gen(z_head).detach()
    h_emb_dup = h_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    r_emb_dup = r_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    t_emb_dup = t_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    h_emb = torch.cat([h_emb_dup, neg_gen_head], dim=1).view(-1, emb_dim)
    r_emb = torch.cat([r_emb_dup, r_emb_dup], dim=1).view(-1, emb_dim)
    t_emb = torch.cat([neg_gen_tail, t_emb_dup], dim=1).view(-1, emb_dim)
    return h_emb, r_emb, t_emb


def get_rank(pos, pred, already_dict, flag):
    if flag == 'tail':
        try:
            already = already_dict[(pos[0, 0].item(), pos[0, 1].item())]
        except:
            already = None
    elif flag == 'head':
        try:
            already = already_dict[(pos[0, 2].item(), pos[0, 1].item())]
        except:
            already = None
    else:
        raise ValueError
    ranking = torch.argsort(pred, descending=True)
    if flag == 'tail':
        rank = (ranking == pos[0, 2]).nonzero().item() + 1
    elif flag == 'head':
        rank = (ranking == pos[0, 0]).nonzero().item() + 1
    else:
        raise ValueError
    ranking_better = ranking[:rank - 1]
    if already != None:
        for e in already:
            if (ranking_better == e).sum() == 1:
                rank -= 1
    return rank


def evaluate(dataloader, already_dict, emb_model, dis, device, cfg, flag):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    with torch.no_grad():
        if cfg.verbose == 1:
            dataloader = tqdm.tqdm(dataloader)
        for pos, X, in dataloader:
            X = X.to(device).squeeze(0)
            h_emb, r_emb, t_emb = emb_model(X)
            pred, _ = dis(h_emb, r_emb, t_emb)
            rank = get_rank(pos, pred, already_dict, flag)
            r.append(rank)
            rr.append(1 / rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    return [r, rr, h1, h3, h10]


def evaluate_wrapper(dataloader_tail, dataloader_head, \
                     already_ts_dict_all, already_hs_dict_all, emb_model, dis, device, cfg, require='tail'):
    if require == 'head':
        head_results = evaluate(dataloader_head, already_hs_dict_all, emb_model, dis, device, cfg, flag='head')
        r = int(sum(head_results[0]) / len(head_results[0]))
        rr = round(sum(head_results[1]) / len(head_results[1]), 4)
        h1 = round(sum(head_results[2]) / len(head_results[2]), 4)
        h3 = round(sum(head_results[3]) / len(head_results[3]), 4)
        h10 = round(sum(head_results[4]) / len(head_results[4]), 4)
    elif require == 'tail':
        tail_results = evaluate(dataloader_tail, already_ts_dict_all, emb_model, dis, device, cfg, flag='tail')
        r = int(sum(tail_results[0]) / len(tail_results[0]))
        rr = round(sum(tail_results[1]) / len(tail_results[1]), 4)
        h1 = round(sum(tail_results[2]) / len(tail_results[2]), 4)
        h3 = round(sum(tail_results[3]) / len(tail_results[3]), 4)
        h10 = round(sum(tail_results[4]) / len(tail_results[4]), 4)
    elif require == 'both':
        tail_results = evaluate(dataloader_tail, already_ts_dict_all, emb_model, dis, device, cfg, flag='tail')
        head_results = evaluate(dataloader_head, already_hs_dict_all, emb_model, dis, device, cfg, flag='head')
        r = int((sum(tail_results[0]) + sum(head_results[0])) / (len(tail_results[0]) * 2))
        rr = round((sum(tail_results[1]) + sum(head_results[1])) / (len(tail_results[1]) * 2), 4)
        h1 = round((sum(tail_results[2]) + sum(head_results[2])) / (len(tail_results[2]) * 2), 4)
        h3 = round((sum(tail_results[3]) + sum(head_results[3])) / (len(tail_results[3]) * 2), 4)
        h10 = round((sum(tail_results[4]) + sum(head_results[4])) / (len(tail_results[4]) * 2), 4)
    else:
        raise ValueError
    print(r, flush=True)
    print(rr, flush=True)
    print(h1, flush=True)
    print(h3, flush=True)
    print(h10, flush=True)
    return rr, h1, h3, h10


def pur_loss11(pred, prior):
    # 传统自对抗
    w = F.softmax(0.5 * pred[:, 1:].clone().detach(), dim=1)
    loss = -F.logsigmoid(pred[:, 0]) - (w * F.logsigmoid(-pred[:, 1:])).sum(dim=1)

    loss = loss.mean()
    return loss


def pur_loss(pred, prior):
    # 传统自对抗 + 高分过滤
    neg_score = pred[:, 1:]
    w = torch.nn.functional.softmax(1.0 * neg_score, dim=1).detach()
    jia = (neg_score >= 0.85 * pred[:, 0].view(neg_score.shape[0], -1).clone().expand_as(neg_score)).float().detach()
    jia *= (1.0 - 2 * w)
    w += jia
    w = torch.nn.functional.normalize(w, p=1, dim=1)
    neg_score = -(w * torch.nn.functional.logsigmoid(-neg_score + 3)).sum(dim=1)
    u = (-torch.nn.functional.logsigmoid(pred[:, 0] - 0).view(neg_score.shape[0]) + neg_score).mean()

    return u


def my_collate_fn(batch):
    return torch.cat(batch, dim=0)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 e_dict,
                 r_dict,
                 train_data,
                 already_ts_dict,
                 already_hs_dict,
                 num_ng):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(train_data.values)
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict
        self.num_ng = num_ng

    def sampling(self, head, rel, tail):
        already_ts = torch.tensor(self.already_ts_dict[(head.item(), rel.item())])
        already_hs = torch.tensor(self.already_hs_dict[(tail.item(), rel.item())])
        neg_pool_t = torch.ones(len(self.e_dict))
        neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero()
        neg_pool_h = torch.ones(len(self.e_dict))
        neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero()
        neg_t = neg_pool_t[torch.randint(len(neg_pool_t), (self.num_ng // 2,))]
        neg_h = neg_pool_h[torch.randint(len(neg_pool_h), (self.num_ng // 2,))]
        return neg_t, neg_h

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        neg_t, neg_h = self.sampling(head, rel, tail)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng // 2, -1), neg_t], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng // 2, -1)], dim=1)
        sample = torch.cat([torch.tensor([head, rel, tail]).unsqueeze(0), neg_t, neg_h], dim=0)
        return sample


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,
                 e_dict,
                 r_dict,
                 test_data,
                 same_entitys_union_dict,
                 same_entitys_intersection_dict,
                 flag):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(test_data.values)
        self.all_e = torch.arange(len(e_dict)).unsqueeze(-1)
        self.flag = flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        if self.flag == 'tail':
            return self.data[idx], torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e],
                                             dim=1)
            if len(same_entitys_intersection_dict[head.item()]) > 0:
                h = torch.tensor(same_entitys_intersection_dict[head.item()]).view(-1, 1)
                rt = torch.tensor([rel, tail]).view(-1, 2).expand(h.shape[0], 2)
                y = torch.cat([h, rt], dim=-1)
            else:
                y = torch.tensor([head, rel, tail]).view(-1, 3)

            return self.data[idx], torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e],
                                             dim=1), y
        elif self.flag == 'head':
            self.data[idx], torch.cat([self.all_e, torch.tensor([rel, tail]).expand(len(self.e_dict), -1)],
                                      dim=1)
            if len(same_entitys_intersection_dict[head.item()]) > 0:
                k = 5
                if len(same_entitys_intersection_dict[head.item()]) <= k:
                    k = len(same_entitys_intersection_dict[head.item()])

                h = [item for item in same_entitys_intersection_dict[head.item()][0:k] for _ in range(len(self.e_dict))]
                h = torch.tensor(h)

                r = torch.tensor([rel]).expand(len(self.e_dict) * k, -1)

                x = k
                number = len(self.e_dict)
                t = [i for _ in range(x) for i in range(0, number)]
                t = torch.tensor(t)

                y = torch.cat([h.view(-1, 1), r.view(-1, 1), t.view(-1, 1)], dim=-1)
            else:
                y = torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e], dim=1)

            return self.data[idx], torch.cat([self.all_e, torch.tensor([rel, tail]).expand(len(self.e_dict), -1)],
                                             dim=1), y
        else:
            raise ValueError


class LookupEmbedding(torch.nn.Module):
    def __init__(self, e_dict, r_dict, emb_dim, bs):
        super().__init__()
        self.emb_dim = emb_dim
        self.bs = bs
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.emb_e = torch.nn.Embedding(len(e_dict), self.emb_dim)
        if (cfg.model == "RotatE"):
            self.emb_e = torch.nn.Embedding(len(e_dict), self.emb_dim * 2)
        self.emb_r = torch.nn.Embedding(len(r_dict), self.emb_dim)
        torch.nn.init.xavier_uniform_(self.emb_e.weight.data)
        torch.nn.init.xavier_uniform_(self.emb_r.weight.data)

    def forward(self, x):
        h, r, t = x[:, 0], x[:, 1], x[:, 2]
        h_emb, r_emb, t_emb = self.emb_e(h), self.emb_r(r), self.emb_e(t)
        return h_emb, r_emb, t_emb

def find_top_k_similar_tensors(a, matrix, k):
    # 将张量和矩阵移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.to(device)
    matrix = matrix.to(device)

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(matrix, a.expand(matrix.size(0), -1), dim=1)


    # 获取前k个相似度最高的索引
    top_k_indices = cos_sim.argsort(descending=True)[:k]

    # 获取相似度最高的前k个张量
    top_k_tensors = matrix[top_k_indices]

    return top_k_tensors

# RotatE
class DistMult(torch.nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()
        self.gamma = torch.tensor(6.0)
        self.epsilon = torch.tensor(2.0)
        self.embedding_range = torch.nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / cfg.emb_dim]),
            requires_grad=False
        )

    def forward(self, h_emb, r_emb, t_emb):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(h_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t_emb, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = r_emb / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=-1)

        score = self.gamma.item() - score.sum(dim=0)
        l2_reg = torch.mean(h_emb ** 2) + torch.mean(t_emb ** 2) + torch.mean(r_emb ** 2)
        return score, l2_reg


class Generator(torch.nn.Module):
    def __init__(self, bs, emb_dim, gen_drop):
        super().__init__()
        self.bs = bs
        self.emb_dim = emb_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim // 8, self.emb_dim // 8),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(gen_drop),
            torch.nn.Linear(self.emb_dim // 8, self.emb_dim),
            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.model(z.view(-1, self.emb_dim // 8)).view(self.bs, -1, self.emb_dim)


def save_score(epoch, rr, h1, h3, h10):
    with open("data.txt", "a") as file:
        # 将数据追加到文件
        txt = str(epoch) + '\t' + str(rr) + '\t' + str(h1) + '\t' + str(h3) + '\t' + str(h10) + '\n'
        txt = "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(epoch, rr, h1, h3, h10)
        file.write(txt)


# 获取相似的实体
def get_same_entitys(test_data, already_ts_dict, already_hs_dict):
    entitys = set(test_data['h'].values)
    same_entitys_union_dict = {}
    same_entitys_intersection_dict = {}
    for entity in entitys:
        same_ts_entitys = [value for value in already_ts_dict.values() if entity in value]
        same_hs_entitys = [value for value in already_hs_dict.values() if entity in value]
        combine = same_hs_entitys + same_ts_entitys
        # 去除重复的值
        combine_set = set(tuple(sublist) for sublist in combine)
        same_entitys = [list(sublist) for sublist in combine_set]
        if len(same_entitys) > 1:
            union = list(set().union(*same_entitys))
            intersection = list(set(same_entitys[0]).intersection(*same_entitys[1:]))
        else:
            if len(same_entitys) == 0:
                union = intersection = []
            else:
                union = intersection = same_entitys[0]

        # 删掉交并集中的实体
        if entity in union:
            union.remove(entity)
        if entity in intersection:
            intersection.remove(entity)
        same_entitys_union_dict[entity] = union
        same_entitys_intersection_dict[entity] = intersection

    return same_entitys_union_dict, same_entitys_intersection_dict


if __name__ == '__main__':
    # preparation
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    # load data
    e_dict, r_dict, train_data, valid_data, test_data, \
        already_ts_dict, already_hs_dict, already_ts_dict_all, already_hs_dict_all = read_data(cfg.data_root + '/')
    # 数据处理，返回：并集 和 交集
    same_entitys_union_dict, same_entitys_intersection_dict = get_same_entitys(test_data, already_ts_dict,
                                                                               already_hs_dict)

    train_dataset = TrainDataset(e_dict, r_dict, train_data, already_ts_dict, already_hs_dict, cfg.num_ng)

    valid_dataset_tail = TestDataset(e_dict, r_dict, valid_data, same_entitys_union_dict,
                                     same_entitys_intersection_dict, flag='tail')
    valid_dataset_head = TestDataset(e_dict, r_dict, valid_data, same_entitys_union_dict,
                                     same_entitys_intersection_dict, flag='head')
    test_dataset_tail = TestDataset(e_dict, r_dict, test_data, same_entitys_union_dict, same_entitys_intersection_dict,
                                    flag='tail')
    test_dataset_head = TestDataset(e_dict, r_dict, test_data, same_entitys_union_dict, same_entitys_intersection_dict,
                                    flag='head')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.bs,
                                                   num_workers=0,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=my_collate_fn)
    valid_dataloader_tail = torch.utils.data.DataLoader(dataset=valid_dataset_tail,
                                                        batch_size=1,
                                                        num_workers=0,
                                                        shuffle=False,
                                                        drop_last=False)
    valid_dataloader_head = torch.utils.data.DataLoader(dataset=valid_dataset_head,
                                                        batch_size=1,
                                                        num_workers=0,
                                                        shuffle=False,
                                                        drop_last=False)
    test_dataloader_tail = torch.utils.data.DataLoader(dataset=test_dataset_tail,
                                                       batch_size=1,
                                                       num_workers=0,
                                                       shuffle=False,
                                                       drop_last=False)
    test_dataloader_head = torch.utils.data.DataLoader(dataset=test_dataset_head,
                                                       batch_size=1,
                                                       num_workers=0,
                                                       shuffle=False,
                                                       drop_last=False)

    # define model
    emb_model = LookupEmbedding(e_dict, r_dict, cfg.emb_dim, cfg.bs)
    dis = DistMult()
    emb_model = emb_model.to(device)
    dis = dis.to(device)

    # define optimizer
    optim_dis = torch.optim.Adam(list(emb_model.parameters()) + list(dis.parameters()), lr=cfg.lrd)

    max_value = 0
    max_score = 0.0

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:', flush=True)
        emb_model.train()
        dis.train()
        avg_loss_dis = []
        avg_loss_gen = []
        if cfg.verbose == 1:
            train_dataloader = tqdm.tqdm(train_dataloader)
        for X in train_dataloader:
            X = X.to(device)
            # ==============================Train D
            #    Train D,Train D,Train D,Train D
            # ==============================Train D
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, reg_real = dis(h_emb, r_emb, t_emb)
            pred = pred_real.view(cfg.bs, -1)
            loss_dis = pur_loss(pred, 0.00001) + cfg.reg * reg_real
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()
            avg_loss_dis.append(loss_dis.item())
        print(f'D Loss: {round(sum(avg_loss_dis) / len(avg_loss_dis), 4)}', flush=True)

        if (epoch + 1) % cfg.valid_interval == 0:
            emb_model.eval()
            dis.eval()
            rr, h1, h3, h10 = evaluate_wrapper(test_dataloader_tail, test_dataloader_head, \
                                               already_ts_dict_all, already_hs_dict_all, emb_model, dis, device, cfg)
            curr_socre = rr * 3 + (h1 + h3 + h10)
            if (max_score < curr_socre):
                max_score = curr_socre
                save_score(epoch + 1, rr, h1, h3, h10)
