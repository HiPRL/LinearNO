import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from torch.utils.tensorboard import SummaryWriter
from model.LinearAttnNeuralOperator import LinearAttentionNeuralOperator
parser = argparse.ArgumentParser('Training')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--key_ratio', type=int, default=4)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='airfoil_LinearAttnNeuralOperator')
parser.add_argument('--data_path', type=str, default='/data/fno/airfoil/naca')
parser.add_argument("--model",type=str,default='None')
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

def count_parameters1(model):
    print(f"{'Module':40s} | {'Params':>10s}")
    print("-" * 55)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            print(f"{name:40s} | {num:10d}")
            total += num
    print("-" * 55)
    print(f"{'Total Trainable Params':40s} | {total:10d}")
    return total


def main():
    # set_seed(0)
    INPUT_X = args.data_path + '/NACA_Cylinder_X.npy'
    INPUT_Y = args.data_path + '/NACA_Cylinder_Y.npy'
    OUTPUT_Sigma = args.data_path + '/NACA_Cylinder_Q.npy'

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    ntrain =1000 
    ntest = 200
    print(ntrain)
    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        x_test, x_test, y_test),
                                              batch_size=1,
                                              shuffle=False)

    print("Dataloading is over.")
    print(x_train.shape,x_test.shape)
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
                  unified_pos=args.unified_pos,
                  H=s1,
                  W=s2,
                  args=args).cuda()
    print(args)
    print(model)
    count_parameters1(model.blocks[0].Attn)
    count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader))
    #l2 LOSS
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

                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.pcolormesh(
                        x[0, :,
                          0].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        x[0, :,
                          1].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        np.zeros([140, 35]),
                        shading='auto',
                        edgecolors='black',
                        linewidths=0.1)
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "input_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(
                        x[0, :,
                          0].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        x[0, :,
                          1].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        out[0, :].reshape(
                            221, 51)[40:180, :35].detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "pred_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(
                        x[0, :,
                          0].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        x[0, :,
                          1].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        y[0, :].reshape(
                            221, 51)[40:180, :35].detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "gt_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(
                        x[0, :,
                          0].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        x[0, :,
                          1].reshape(221,
                                     51)[40:180, :35].detach().cpu().numpy(),
                        out[0, :].reshape(
                            221, 51)[40:180, :35].detach().cpu().numpy() -
                        y[0, :].reshape(
                            221, 51)[40:180, :35].detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             "error_" + str(id) + ".pdf"),
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))

    else:
        writer = SummaryWriter(log_dir='./runs/' + args.save_name)
        for ep in tqdm(range(args.epochs)):

            model.train()
            train_loss = 0

            for pos, fx, y in tqdm(train_loader):

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda(
                )  # x:B,N,2  fx:B,N,2  y:B,N
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                # print(out.shape, y.shape)
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
            writer.add_scalars('loss', {"train_loss": train_loss}, ep)

            if (ep) % 10 == 0 or ep == args.epochs - 1:
                model.eval()
                rel_err = 0.0
                with torch.no_grad():
                    for pos, fx, y in test_loader:
                        x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                        out = model(x, None).squeeze(-1)

                        tl = myloss(out, y).item()
                        rel_err += tl

                rel_err /= ntest
                tqdm.write("rel_err:{}".format(rel_err))
                writer.add_scalars('loss', {"rel_err": rel_err}, ep)

            if (ep + 1) % 100 == 0:
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
