import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from model.LinearAttnNeuralOperator import LinearAttentionNeuralOperator
from tqdm import tqdm

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--key_ratio', type=int, default=4)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='elas_LinearAttnNeuralOperator')
parser.add_argument('--data_path', type=str, default='/data/fno')
parser.add_argument("--model",type=str,default='2D_temp')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    torch.backends.cudnn.deterministic = True
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    ntrain = args.ntrain
    ntest = 200
    # set_seed(0)
    PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    print(input_s.shape, input_xy.shape)

    y_normalizer = UnitTransformer(train_s)

    train_s = y_normalizer.encode(train_s)
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        train_xy, train_xy, train_s),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        test_xy, test_xy, test_s),
                                              batch_size=1,
                                              shuffle=False)

    print("Dataloading is over.")
    print(train_xy.shape,test_xy.shape)
    model = LinearAttentionNeuralOperator(space_dim=2,
                    n_layers=args.n_layers,
                    n_hidden=args.n_hidden,
                    dropout=args.dropout,
                    n_head=args.n_heads,
                    Time_Input=False,
                    mlp_ratio=args.mlp_ratio,
                    fun_dim=0,
                    out_dim=1,
                    key_ratio=args.key_ratio,
                    ref=args.ref,
                    unified_pos=args.unified_pos,args=args).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)


    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"),strict=True)
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        rel_err = 0.0
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(),
                                y=fx[0, :, 1].detach().cpu().numpy(),
                                c=y[0, :].detach().cpu().numpy(),
                                cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "gt_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(),
                                y=fx[0, :, 1].detach().cpu().numpy(),
                                c=out[0, :].detach().cpu().numpy(),
                                cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "pred_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(),
                                y=fx[0, :, 1].detach().cpu().numpy(),
                                c=((y[0, :] -
                                    out[0, :])).detach().cpu().numpy(),
                                cmap='coolwarm')
                    plt.clim(-8, 8)
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "error_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err : {}".format(rel_err))
    else:
        for ep in tqdm(range(args.epochs)):

            model.train()
            train_loss = 0

            for pos, fx, y in tqdm(train_loader):

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda(
                )  # x:B,N,2  fx:B,N,2  y:B,N,
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            train_loss = train_loss / ntrain
            tqdm.write("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            tqdm.write("rel_err : {}".format(rel_err))

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                tqdm.write('save model')
                torch.save(model.state_dict(),
                           os.path.join('./checkpoints', save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(),
                   os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    main()
