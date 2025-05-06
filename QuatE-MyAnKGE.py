import os
from collections import Counter

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
    parser.add_argument('--model', default='QuatE', type=str)  # 模型
    parser.add_argument('--num_ng', default=32, type=int)  # 负样本个数
    parser.add_argument('--bs', default=512, type=int)  # batch_size
    parser.add_argument('--emb_dim', default=1024, type=int)  # 维度
    parser.add_argument('--lrd', default=0.0001, type=float)  # 学习率
    parser.add_argument('--reg', default=0.001, type=float)  # 正则
    parser.add_argument('--refer', default=0.5, type=float)  # 参考答案比例
    parser.add_argument('--refer_num', default=4, type=int)  # 参考答案个数

    # Misc
    open_set = 1
    finally_set = './data/FB15k-237' if open_set == 1 else './data/WN18RR'
    parser.add_argument('--data_root', default=finally_set, type=str)  # 数据集
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--seed', default=42, type=int)  # 随机数种子
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)  # 一共训练多少轮
    parser.add_argument('--valid_interval', default=2, type=int)  # 每隔几轮测试一次
    return parser.parse_args(args)


def save_dataset(filename, data):
    path = "./dataset/"
    if not os.path.exists(path):
        os.makedirs(path)
    if filename.split('.')[-1] == 'txt':
        with open(path + filename, 'w', encoding='utf-8') as file:
            file.write(str(data))
    else:
        pf = pd.DataFrame(data)
        pf.to_csv(path + filename, index=False)


def read_dataset(filename):
    path = "./dataset/"
    if filename.split('.')[-1] == 'txt':
        with open(path + filename, 'r') as file:
            data = eval(file.read())
    else:
        data = pd.read_csv(path + filename)
    return data


def file_is_exist(filename):
    path = "./dataset/"
    return os.path.exists(path + filename)


def read_data(root):
    # read entity dict and relation dict\
    # save_file = ['e_dict.txt', 'r_dict.txt', 'train_data.csv', 'valid_data.csv', 'test_data.csv', 'already_ts_dict.txt',
    #              'already_hs_dict.txt', 'already_rs_dict.txt', 'already_ts_dict_all.txt', 'already_hs_dict_all.txt',
    #              'already_rs_dict_all.txt']

    if file_is_exist('already_rs_dict_all.txt'):  # 简单判断,最后一个文件存在,默认其他都存在
        # 开始加载数据

        e_dict = read_dataset('e_dict.txt')

        r_dict = read_dataset('r_dict.txt')
        train_data = read_dataset('train_data.csv')
        valid_data = read_dataset('valid_data.csv')
        test_data = read_dataset('test_data.csv')
        already_ts_dict = read_dataset('already_ts_dict.txt')
        already_hs_dict = read_dataset('already_hs_dict.txt')
        already_rs_dict = read_dataset('already_rs_dict.txt')
        already_ts_dict_all = read_dataset('already_ts_dict_all.txt')
        already_hs_dict_all = read_dataset('already_hs_dict_all.txt')
        already_rs_dict_all = read_dataset('already_rs_dict_all.txt')

        return (e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict, already_rs_dict,
                already_ts_dict_all, already_hs_dict_all, already_rs_dict_all)
    else:  # 数据不存在,重新加载
        e_dict = {}
        r_dict = {}
        e_data = pd.read_csv(root + 'entities.dict', header=None, delimiter='\t').values
        r_data = pd.read_csv(root + 'relations.dict', header=None, delimiter='\t').values
        for record in r_data:
            r_dict[record[1]] = record[0]
        for record in e_data:
            e_dict[record[1]] = record[0]
        save_dataset('e_dict.txt', e_dict)
        save_dataset('r_dict.txt', r_dict)
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

        save_dataset('train_data.csv', train_data)
        save_dataset('valid_data.csv', valid_data)
        save_dataset('test_data.csv', test_data)

        # already existing heads or tails (for sampling and evaluation)
        already_ts_dict = {}
        already_hs_dict = {}
        already_rs_dict = {}
        already_ts = train_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
        already_hs = train_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
        already_rs = train_data.groupby(['h', 't'])['r'].apply(list).reset_index(name='rs').values
        for record in already_ts:
            already_ts_dict[(record[0], record[1])] = record[2]
        for record in already_hs:
            already_hs_dict[(record[0], record[1])] = record[2]
        for record in already_rs:
            already_rs_dict[(record[0], record[1])] = record[2]

        save_dataset('already_ts_dict.txt', already_ts_dict)
        save_dataset('already_hs_dict.txt', already_hs_dict)
        save_dataset('already_rs_dict.txt', already_rs_dict)

        all_data = pd.concat([train_data, valid_data, test_data])
        already_ts_dict_all = {}
        already_hs_dict_all = {}
        already_rs_dict_all = {}
        already_ts_all = all_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
        already_hs_all = all_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
        already_rs_all = all_data.groupby(['h', 't'])['r'].apply(list).reset_index(name='rs').values
        for record in already_ts_all:
            already_ts_dict_all[(record[0], record[1])] = record[2]
        for record in already_hs_all:
            already_hs_dict_all[(record[0], record[1])] = record[2]
        for record in already_rs_all:
            already_rs_dict_all[(record[0], record[1])] = record[2]

    save_dataset('already_ts_dict_all.txt', already_ts_dict_all)
    save_dataset('already_hs_dict_all.txt', already_hs_dict_all)
    save_dataset('already_rs_dict_all.txt', already_rs_dict_all)

    return (e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict, already_rs_dict,
            already_ts_dict_all, already_hs_dict_all, already_rs_dict_all)


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
        for pos, X, ankge in dataloader:

            X = X.to(device).squeeze(0)
            ankge = ankge.to(device).squeeze(0)
            h_emb, r_emb, t_emb = emb_model(X)  # print(h_emb.shape)  40943 * 2048
            h_emb_an, _, _ = emb_model(ankge)  # print(h_emb.shape)  n * 2048

            # 获取真实三元组中头实体的嵌入向量
            true_h_emb = pos.clone().to(device)
            true_h_emb, _, _ = emb_model(true_h_emb)

            # 示例用法

            h_emb_an = find_top_k_similar_tensors(true_h_emb, h_emb_an, cfg.refer_num)
            sim_score = (true_h_emb * h_emb_an).sum(dim=-1)
            sim_score = F.softmax(sim_score, dim=0).view(len(sim_score), -1)
            sim_score = 1 - sim_score
            sim_score = torch.nn.functional.normalize(sim_score, p=1, dim=0)

            t_emb_an = t_emb.repeat(h_emb_an.shape[0], 1)
            r_emb_an = r_emb.repeat(h_emb_an.shape[0], 1)
            # 复制每一行n次
            h_emb_an = torch.cat([torch.unsqueeze(h_emb_an, 1)] * h_emb.shape[0], dim=1).view(-1, h_emb_an.shape[1])

            pred, _ = dis(h_emb, r_emb, t_emb)
            pred_an, _ = dis(h_emb_an, r_emb_an, t_emb_an)
            pred_an = pred_an.view(-1, pred.shape[-1]).mean(dim=0)
            pred_an = (sim_score * pred_an).sum(dim=0)
            pred += cfg.refer * pred_an

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
    neg_score = -(w * torch.nn.functional.logsigmoid(-neg_score)).sum(dim=1)
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
                 same_entities,
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
            if len(same_entities[head.item()]) > 0:
                h = torch.tensor(same_entities[head.item()]).view(-1, 1)
                rt = torch.tensor([rel, tail]).view(-1, 2).expand(h.shape[0], 2)
                y = torch.cat([h, rt], dim=-1)
            else:
                y = torch.tensor([head, rel, tail]).view(-1, 3)

            return self.data[idx], torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e],
                                             dim=1), y
        elif self.flag == 'head':
            if len(same_entities[head.item()]) > 0:
                k = 5
                if len(same_entities[head.item()]) <= k:
                    k = len(same_entities[head.item()])

                h = [item for item in same_entities[head.item()][0:k] for _ in
                     range(len(self.e_dict))]
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

# QuatE
class DistMult(torch.nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()
        self.gamma = torch.tensor(9.0)
        self.epsilon = torch.tensor(2.0)
        self.embedding_range = torch.nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / cfg.emb_dim]),
            requires_grad=False
        )

    def forward(self, h_emb, r_emb, t_emb):
        s_h, x_h, y_h, z_h = torch.chunk(h_emb, 4, dim=-1)
        s_r, x_r, y_r, z_r = torch.chunk(r_emb, 4, dim=-1)
        s_t, x_t, y_t, z_t = torch.chunk(t_emb, 4, dim=-1)

        denominator_r = torch.sqrt(s_h ** 2 + x_h ** 2 + y_h ** 2 + z_h ** 2)

        s_r = s_r / denominator_r
        x_r = x_r / denominator_r
        y_r = y_r / denominator_r
        z_r = z_r / denominator_r


        A = s_h * s_r - x_h * x_r - y_h * y_r - z_h * z_r
        B = s_h * x_r + s_r * x_h + y_h * z_r - y_r * z_h
        C = s_h * y_r + s_r * y_h + z_h * x_r - z_r * x_h
        D = s_h * z_r + s_r * z_h + x_h * y_r - x_r * y_h

        score = (A * s_t + B * x_t + C * y_t + D * z_t)
        score = score.sum(dim=-1)

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


def get_same_entities(test_data, already_ts_dict, already_hs_dict):
    entities = set(test_data['h'].values)
    same_entities = {}
    print("开始处理实体:")
    for entity in tqdm.tqdm(entities):
        # 在dict1中查找包含num的列表
        dict1_lists = [value for value in already_hs_dict.values() if entity in value]

        # 在dict2中查找包含num的列表
        dict2_lists = [value for value in already_ts_dict.values() if entity in value]

        # 合并列表
        merged_list = [item for sublist in dict1_lists + dict2_lists for item in sublist]
        if len(set(merged_list)) != len(merged_list):
            # 统计数字出现的次数
            count = Counter(merged_list)
            # 删除出现次数小于1次的数字
            filtered_list = [key for key, value in count.items() if value > 1]
        else:
            filtered_list = merged_list
        if entity in filtered_list:
            filtered_list.remove(entity)
        # 更新结果字典
        same_entities[entity] = filtered_list
    save_dataset('same_entities.txt', same_entities)
    return same_entities



def get_same_relations(test_data, already_rs_dict):
    relations = set(test_data['r'].values)
    # 过滤掉值中长度为1的数据，这些数据没有意义
    data = {key: values for key, values in already_rs_dict.items() if len(values) >= 2}
    same_relations = {}
    print("开始处理关系:")
    for relation in tqdm.tqdm(relations):
        result = []
        # 找到数据中存在且数量大于1的数
        for values in data.values():
            if relation in values and len(values) > 1:
                result.extend(values)
            # 特殊情况数据只有[1,2]防止被后面过滤掉
        if len(set(result)) != len(result):
            result = list(set([element for element in result if result.count(element) > 1]))
        if relation in result:
            result.remove(relation)
        same_relations[relation] = result
    save_dataset('same_relations.txt', same_relations)
    return same_relations


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
        already_ts_dict, already_hs_dict, already_rs_dict, \
        already_ts_dict_all, already_hs_dict_all, already_rs_dict_all = read_data(cfg.data_root + '/')
    # 数据处理，返回：并集 和 交集
    # 数据保存过直接加载数据
    if file_is_exist('same_entities.txt'):
        same_entities = read_dataset('same_entities.txt')
    else:
        # 重新加载数据
        same_entities = get_same_entities(test_data, already_ts_dict, already_hs_dict)
    # 数据存在直接加载数据
    if file_is_exist('same_relations.txt'):
        same_relation = read_dataset('same_relations.txt')
    else:
        # 数据不存在重新加载数据
        same_relation = get_same_relations(test_data, already_rs_dict)

    train_dataset = TrainDataset(e_dict, r_dict, train_data, already_ts_dict, already_hs_dict, cfg.num_ng)

    valid_dataset_tail = TestDataset(e_dict, r_dict, valid_data, same_entities, flag='tail')
    valid_dataset_head = TestDataset(e_dict, r_dict, valid_data, same_entities, flag='head')
    test_dataset_tail = TestDataset(e_dict, r_dict, test_data,same_entities,
                                    flag='tail')
    test_dataset_head = TestDataset(e_dict, r_dict, test_data, same_entities,
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
            emb_model.train()
            dis.train()
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, reg_real = dis(h_emb, r_emb, t_emb)
            pred = pred_real.view(cfg.bs, -1)
            loss_dis = pur_loss(pred, 0.00001) + cfg.reg * 0.5 * reg_real
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
