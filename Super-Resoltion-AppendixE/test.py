import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from module.utils import *

parser = argparse.ArgumentParser(description="Latent Neural Operator PyTorch")
parser.add_argument('--config', type=str, default=None, required=True)
parser.add_argument('--device', type=str, default=None, required=True)
parser.add_argument('--seed',  type=int, default=0)
args = parser.parse_args()


def test_completer(test_dataloader,
        transformer,
        masker,
        poser,
        model,
        model_name,
        local_rank,
        world_size):

    with torch.no_grad():
        device = local_rank
        test_metric = 0
        test_side1_metric = 0
        test_side2_metric = 0

        for _, data in enumerate(test_dataloader):
            x, y, _ = data
            if "DeepONet" in model_name:
                x, y, ob = data_preprocess_completer_DeepONet(masker, x, y, device)
            elif "GNOT" in model_name:
                x, y, ob = data_preprocess_completer_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_completer_LNO(masker, x, y, device)
            elif "LinearNO" in model_name:
                x, y, ob = data_preprocess_completer_LNO(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Completer Name")

            model.eval()
            res = model(x, ob)
            mask, vertex0, vertex1 = masker.get()
            res = torch.reshape(res, (res.shape[0], *tuple(vertex1 - vertex0), res.shape[-1]))
            y = torch.reshape(y, (y.shape[0], *tuple(vertex1 - vertex0), y.shape[-1]))

            res = transformer.apply_y(res, inverse=True)
            y = transformer.apply_y(y, inverse=True)

            pos = poser.get()
            pos_idx = list(pos.to_sparse().indices().transpose(0,1).numpy())
            res = res.cpu().numpy()
            res = torch.tensor(np.array([res[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)
            y = y.cpu().numpy()
            y = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)

            p = 2
            metric = RelLpLoss(p)
            test_batch_metric = torch.tensor(metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_metric)
            test_metric = test_metric + test_batch_metric / world_size

            p = 2
            side1_metric = MpELoss(p)
            test_batch_side1_metric = torch.tensor(side1_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side1_metric)
            test_side1_metric = test_side1_metric + test_batch_side1_metric / world_size

            p = 1
            side2_metric = RelMpELoss(p)
            test_batch_side2_metric = torch.tensor(side2_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side2_metric)
            test_side2_metric = test_side2_metric + test_batch_side2_metric / world_size

        test_metric /= len(test_dataloader)
        test_side1_metric /= len(test_dataloader)
        test_side2_metric /= len(test_dataloader)

    return test_metric, test_side1_metric, test_side2_metric


if __name__ == "__main__":
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.distributed.init_process_group("nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = local_rank
    torch.cuda.set_device(device)

    config_file = "./configs/" + args.config + ".jsonc"
    config = Configuration(config_file)

    train_dataset, train_sampler, train_dataloader, \
    val_dataset, val_sampler, val_dataloader, \
    test_dataset, test_sampler, test_dataloader, \
    transformer, masker, poser, \
    model, loss, optimizer, scheduler \
    = get_data_model(config, device)
    
    log_dir = "./experiments/" + args.config + "/log/"
    checkpoint_dir = "./experiments/" + args.config + "/checkpoint/"
    model.load_state_dict(torch.load(checkpoint_dir + "500.pt",  map_location=f"cuda:{device}"))
    if local_rank == 0:
        print(config)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(log_dir + "config.json", "w") as f:
            json.dump(dict(config), f, indent=4)

        test_metric, test_side1_metric, test_side2_metric = test_completer(
            test_dataloader,
            transformer,
            masker,
            poser,
            model,
            config.model.name,
            local_rank,
            world_size,
        )
        print(f"Relative L2 errors: {test_metric:.6f}, MSE: {test_side1_metric:.6f}, Relative MAE: {test_side2_metric:.6f}")
    torch.distributed.destroy_process_group()
