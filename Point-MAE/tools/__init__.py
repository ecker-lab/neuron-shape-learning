# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net
from .runner_cluster import cluster_net as cluster_run_net
from .runner_classification import classify_net as classify_run_net
