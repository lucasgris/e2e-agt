import shutil
import os

for exp_dir in os.listdir():
    if os.path.isdir(exp_dir):
        for y in os.listdir(exp_dir):
            for d in os.listdir(os.path.join(exp_dir, y)):
                if os.path.isdir(os.path.join(exp_dir, y, d)):
                    if os.path.isdir(os.path.join(exp_dir, y, d, "checkpoints")):
                        if len(os.listdir(os.path.join(exp_dir, y, d, "checkpoints"))) == 0:
                            print(os.path.join(exp_dir, y, d, "checkpoints"))
                            shutil.rmtree(os.path.join(exp_dir, y, d))
                    