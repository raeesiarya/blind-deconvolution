import wandb
import os
from dotenv import load_dotenv
from testing.testbench import testebench
from testing.testbench_configs import TESTBENCH_CONFIGS

def main():
    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY") # from https://wandb.ai/authorize
    wandb.login()

    for cfg in TESTBENCH_CONFIGS:
        cfg = dict(cfg)  # shallow copy to avoid mutating the source
        run_name = cfg.pop("name", None)
        testebench(run_name=run_name, **cfg)

if __name__ == "__main__":
    main()
