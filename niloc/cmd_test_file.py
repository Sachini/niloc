import os.path as osp
import sys
from subprocess import Popen

"""
Run all localization tasks for specified checkpoints.
Checkpoint list is given as `+ckpt_file=<path>`, which contains the name: checkpoint pairs to be tested.
Checkpoint is the relative path from file directory.
E.g.
100 version_0/epoch=100-tr_ratio=0.9-val_loss=9.46.ckpt
150 version_0/epoch=150.ckpt
...
"""


tasks = {
    "scheduled_1branch": {
        "start_gt_2": ["+test_cfg.start_gt=2"],
        "start_gt_1": ["+test_cfg.start_gt=1"],
        "start_zero": ["+test_cfg.start_gt=0"],
    },
    "scheduled_2branch": {
        "encoder": ["+test_cfg.with_gt=1", "+test_cfg.encoder_only=True"],
        "start_gt_2": ["+test_cfg.start_gt=2", "+test_cfg.decoder_only=True"],
        "start_gt_1": ["+test_cfg.start_gt=1", "+test_cfg.decoder_only=True"],
        "start_zero": ["+test_cfg.start_gt=0", "+test_cfg.decoder_only=True"],
    },
    "standard_1branch": {"encoder": []},
}


def get_checkpoints(file_name):
    checkpoints = {}
    parent_dir = osp.dirname(file_name)
    with open(file_name) as f:
        for s in f.readlines():
            if len(s.strip()) > 0 and s[0] != '#':
                c = s.split()
                checkpoints[c[0]] = osp.abspath(osp.join(parent_dir, c[1]))
                assert osp.exists(checkpoints[c[0]]), f"Cannot find checkpoint {c[0]}: {checkpoints[c[0]]}"
    print(f"Found {len(checkpoints)} checkpoints")
    print(checkpoints)
    return checkpoints


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    cmds = list(sys.argv[1:])
    task, name_idx, my_file_idx = None, -1, -1
    print_only, linear, gpu = False, False, None
    del_list = []
    for i, arg in enumerate(cmds):
        if arg.startswith("test_cfg.test_name="):
            name_idx = i
        if arg.startswith("task="):
            task = arg.split("=")[1]
        if "test_cfg.model_path=" in arg:
            print(arg)
            cmds[i] = "'" + cmds[i] + "'"
            print("All checkpoints must be in file. No test_cfg.model_path")
            exit(1)
        if arg.startswith("ckpt_file="):
            my_file_idx = i
        if arg.startswith("print_only="):
            del_list.append(i)
            print_only = True
        if arg.startswith("use_gpu="):
            linear = True
            gpu = int(arg.split('=')[1])
            cmds[i] = "train_cfg.gpus=1"
        if arg.startswith("linear="):
            linear = arg.split("=")[1] not in ['False', "False", False] or linear
            del_list.append(i)

    if my_file_idx == -1 or name_idx == -1:
        print("ckpt_file and test_cfg.test_name are required")
    for i in del_list:
        del cmds[i]

    print(task, cmds[name_idx])
    assert task in tasks, "Unknown task"
    cwd = osp.split(osp.dirname(osp.abspath(__file__)))[0] + '/'
    print("directory ", cwd)
    checkpoints = get_checkpoints(cmds[my_file_idx].split("=", 1)[1])
    parent_dir = osp.dirname(cmds[my_file_idx].split("=", 1)[1])


    def get_command(key, ckpt):
        commands = cmds + tasks[task][key]
        commands[name_idx] += f"{ckpt}_{key}"
        commands[my_file_idx] = f"'test_cfg.model_path=\"{checkpoints[ckpt]}\"'"
        cmd_string = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        cmd_string += "python3 niloc/evaluate.py "
        cmd_string += " ".join(commands)
        print(cmd_string)

        return cmd_string


    for k in checkpoints:
        if print_only:
            _ = [get_command(key, k) for key in tasks[task]]
            continue
        elif linear:
            for key in tasks[task]:
                proc = Popen(get_command(key, k), shell=True, cwd=cwd)
                proc.wait()
                print(f"checkpoint {k} done")
        else:
            procs_list = [Popen(get_command(key, k), shell=True, cwd=cwd)
                          for key in tasks[task]]
            for proc in procs_list:
                proc.wait()
            print(f"checkpoint {k} done")
