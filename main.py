import argparse
import os
import random
from os.path import join

import numpy as np
import torch

from trainer import Trainer
from utils.noter import Noter


def main(n):
    parser = argparse.ArgumentParser(description="Code for BIG_LMRec")
    parser.add_argument(
        "--data",
        type=str,
        default="abe",
        help=" afk=Food-Kitchen, amb=Movie-Book, abe=Beauty-Electronics",
    )

    parser.add_argument('--name', type=str, default='BIG_LMRec Model Plus', help='model name')

    # Capsule
    parser.add_argument("--k_max", type=int, default=20, help="The number of layers in the multi-layer structure")
    parser.add_argument("--iters", type=int, default=20, help="The number of iterations in dynamic routing")

    # lora
    parser.add_argument("--rank", type=int, default=2, help="the rank of LoRA")
    parser.add_argument("--num_experts", type=int, default=3, help="The number of expert numbers")
    parser.add_argument("--top_k", type=int, default=2, help="The number of top experts selected")
    parser.add_argument("--layer_num_x", type=int, default=9, help="layer_num_x")
    parser.add_argument("--layer_num_a", type=int, default=7, help="layer_num_a")
    parser.add_argument("--layer_num_b", type=int, default=7, help="layer_num_b")

    # Data
    parser.add_argument(
        "--raw", action="store_true", help="use raw data from c2dsr, takes longer time"
    )
    parser.add_argument(
        "--len_max", type=int, default=50, help="# of interactions allowed to input"
    )
    parser.add_argument("--n_neg", type=int, default=128, help="# negative inference samples")
    parser.add_argument(
        "--n_mtc", type=int, default=999, help="# negative metric samples"
    )

    # Model
    parser.add_argument("--d_latent", type=int, default=256, help="The dimensionality of the latent representation vector")
    parser.add_argument("--n_head", type=int, default=2, help="multi head attention")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--temp", type=float, default=0.75, help="temperature for Recommendation")

    # Training
    parser.add_argument("--cuda", type=str, default="0", help="running device")
    parser.add_argument(
        "--seed", type=int, default=3407, help="radom seed, 3407 is all you need"
    )
    parser.add_argument("--bs", type=int, default=256, help="batch_size")
    parser.add_argument(
        "--n_worker",
        type=int,
        default=28,
        help="# dataloader worker",
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=9999999,
        help="# epoch maximum",
    )
    parser.add_argument("--n_warmup", type=int, default=6, help="# warmup epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--l2", type=float, default=2, help="weight decay"
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.3162,
        help="scheduler gamma",
    )
    parser.add_argument(
        "--lr_p", type=int, default=49, help="scheduler patience"
    )

    args = parser.parse_args()

    if args.cuda == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.cuda}")

    args.len_trim = args.len_max - 3
    args.es_p = (args.lr_p + 1) * 2 - 1

    # paths
    args.path_root = os.getcwd()
    args.path_data = join(args.path_root, "data", args.data)
    args.path_log = join(args.path_root, "log",)
    for p in [args.path_data, args.path_log]:
        if not os.path.exists(p):
            os.makedirs(p)

    args.f_raw = join(args.path_data, f"{args.data}_{args.len_max}_preprocessed.txt")
    args.f_data = join(args.path_data, f"{args.data}_{args.len_max}_seq.pkl")

    if args.raw and not os.path.exists(args.f_raw):
        raise FileNotFoundError(
            f"Selected preprocessed dataset {args.data} does not exist."
        )
    if not args.raw and not os.path.exists(args.f_data):
        if os.path.exists(args.f_raw):
            raise FileNotFoundError(
                f'Selected dataset {args.data} need process, specify "--raw" in the first run.'
            )
        raise FileNotFoundError(
            f"Selected processed dataset {args.data} does not exist."
        )

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # modeling
    noter = Noter(args)
    trainer = Trainer(args, noter)

    cnt_es, cnt_lr, mrr_log = 0, 0, 0.0
    res_f2a, res_f2b, res_c2a, res_c2b, res_a2a, res_a2b, res_b2a, res_b2b = (
        [0] * 4,
        [0] * 4,
        [0] * 4,
        [0] * 4,
        [0] * 4,
        [0] * 4,
        [0] * 4,
        [0] * 4,
    )

    for epoch in range(1, args.n_epoch + 1):
        lr_cur = trainer.optimizer.param_groups[0]["lr"]
        res_val = trainer.run_epoch(epoch)
        mrr_val = res_val[0][-1] + res_val[1][-1]  # use ndcg@10 as identifier
        noter.log_valid(res_val[0], res_val[1])
        if epoch <= args.n_warmup:
            lr_str = f"{lr_cur:.5e}"
            noter.log_msg(f"|     |  lr | {lr_str[:3]}e-{lr_str[-1]} | warmup |")
            trainer.scheduler_warmup.step()
        else:
            if mrr_val >= mrr_log:
                mrr_log = mrr_val
                cnt_es = 0
                cnt_lr = 0
                lr_str = f"{lr_cur:.5e}"
                noter.log_msg(
                    f"|     |  lr | {lr_str[:3]}e-{lr_str[-1]} |  0 /{args.lr_p:2} |  0 /{args.es_p:2} |"
                )

                (
                    res_f2a,
                    res_f2b,
                    res_c2a,
                    res_c2b,
                    res_a2a,
                    res_a2b,
                    res_b2a,
                    res_b2b,
                ) = res_val
                noter.log_test(
                    res_f2a,
                    res_f2b,
                    res_c2a,
                    res_c2b,
                    res_a2a,
                    res_a2b,
                    res_b2a,
                    res_b2b,
                )
                trainer.scheduler.step(epoch)

            else:
                cnt_lr += 1
                cnt_es += 1
                if cnt_es > args.es_p:
                    noter.log_msg(f"\n[info] Exceeds maximum early-stop patience.")
                    break
                else:
                    trainer.scheduler.step(0)

                    lr_str = f"{lr_cur:.5e}"
                    noter.log_msg(
                        f"|     | lr  | {lr_str[:3]}e-{lr_str[-1]} "
                        f"| {cnt_lr:2} /{args.lr_p:2} | {cnt_es:2} /{args.es_p:2} |"
                        f" best mrr | {mrr_log:.5f} | current mrr | {mrr_val:.5f} |"
                    )
                    if lr_cur != trainer.optimizer.param_groups[0]["lr"]:
                        cnt_lr = 0
                if epoch == 50 and mrr_log < 0.2:
                    break
                if epoch == 150 and mrr_log < 0.3:
                    break

    noter.log_final(
        res_f2a, res_f2b, res_c2a, res_c2b, res_a2a, res_a2b, res_b2a, res_b2b
    )
    noter.log_num_param(trainer.model)


if __name__ == "__main__":
    for i in [0]:
        main(i)
