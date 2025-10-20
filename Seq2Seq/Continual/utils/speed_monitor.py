import math
import time
import torch
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional
from lightning import Callback, Fabric, LightningModule, Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only as fabric_rank_zero_only
from lightning.pytorch.utilities.rank_zero import rank_zero_only as trainer_rank_zero_only

GPU_AVAILABLE_FLOPS = {
    "h100-sxm": {
        "64-true": 67e12,
        "32-true": 67e12,
        "16-true": 1.979e15 / 2,
        "16-mixed": 1.979e15 / 2,
        "bf16-true": 1.979e15 / 2,
        "bf16-mixed": 1.979e15 / 2,
        "8-true": 3.958e15 / 2,
        "8-mixed": 3.958e15 / 2,
    },
    "h100-pcie": {
        "64-true": 51e12,
        "32-true": 51e12,
        "16-true": 1.513e15 / 2,
        "16-mixed": 1.513e15 / 2,
        "bf16-true": 1.513e15 / 2,
        "bf16-mixed": 1.513e15 / 2,
        "8-true": 3.026e15 / 2,
        "8-mixed": 3.026e15 / 2,
    },
    "a100": {
        "64-true": 19.5e12,
        "32-true": 19.5e12,
        "16-true": 312e12,
        "16-mixed": 312e12,
        "bf16-true": 312e12,
        "bf16-mixed": 312e12,
    },
    "a10g": {"32-true": 31.2e12, "16-true": 125e12, "16-mixed": 125e12, "bf16-true": 125e12, "bf16-mixed": 125e12},
    "v100-sxm": {"64-true": 7.8e12, "32-true": 15.7e12, "16-true": 125e12, "16-mixed": 125e12},
    "v100-pcie": {"64-true": 7e12, "32-true": 14e12, "16-true": 112e12, "16-mixed": 112e12},
    "v100s-pcie": {"64-true": 8.2e12, "32-true": 16.4e12, "16-true": 130e12, "16-mixed": 130e12},
    "t4": {"32-true": 8.1e12, "16-true": 65e12, "16-mixed": 65e12, "8-true": 130e12, "int4": 260e12},
    "quadro rtx 5000": {"32-true": 11.2e12, "16-true": 89.2e12, "16-mixed": 89.2e12},
}

TPU_AVAILABLE_FLOPS = {
    "v2": 45e12,
    "v3": 123e12,
    "v4": 275e12,
}

def get_flops_available(device: torch.device, precision: str) -> Optional[float]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device).lower()
        if "h100" in device_name and "hbm3" in device_name:
            device_name = "h100-sxm"
        elif "h100" in device_name and ("pcie" in device_name or "hbm2e" in device_name):
            device_name = "h100-pcie"
        elif "a100" in device_name:
            device_name = "a100"
        elif "a10g" in device_name:
            device_name = "a10g"
        elif "v100-sxm" in device_name:
            device_name = "v100-sxm"
        elif "v100-pcie" in device_name:
            device_name = "v100-pcie"
        elif "t4" in device_name:
            device_name = "t4"
        elif "quadro rtx 5000" in device_name:
            device_name = "quadro rtx 5000"
        else:
            device_name = None

        if device_name is not None:
            try:
                return int(GPU_AVAILABLE_FLOPS[device_name][precision])
            except KeyError:
                raise KeyError(
                    f"flop count not found for {device_name} with precision: {precision}; "
                    "MFU cannot be calculated and reported."
                )

    return None


class SpeedMonitorBase:
    
    def __init__(
        self,
        flops_available: float,
        log_dict: Callable[[Dict, int], None],
        window_size: int = 100,
        time_unit: str = "hours",
        log_iter_interval: int = 1,
    ):
        self.flops_available = flops_available
        self.log_dict = log_dict
        self.log_iter_interval = log_iter_interval
        self.history_samples: Deque[int] = deque(maxlen=window_size + 1)
        self.history_training_loss: Deque[int] = deque(maxlen=log_iter_interval)
        self.history_wct: Deque[float] = deque(maxlen=window_size + 1)
        self.history_lengths: Deque[int] = deque(maxlen=window_size + 1)
        self.history_flops: Deque[int] = deque(maxlen=window_size + 1)

        self.divider = 1
        if time_unit == "seconds":
            self.divider = 1
        elif time_unit == "minutes":
            self.divider = 60
        elif time_unit == "hours":
            self.divider = 60 * 60
        elif time_unit == "days":
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".'
            )

        self.total_eval_wct = 0.0
        self.iter = -1

    def on_train_batch_end(
        self,
        samples: int,
        train_elapsed: float,
        world_size: int,
        step_count: int,
        flops_per_batch: Optional[int] = None,
        lengths: Optional[int] = None,
        train_loss: Optional[float] = None,
    ):
        self.iter += 1
        metrics = {}

        self.history_samples.append(samples)
        self.history_training_loss.append(train_loss)
        if lengths is not None:
            self.history_lengths.append(lengths)
            assert len(self.history_samples) == len(self.history_lengths)
        self.history_wct.append(train_elapsed)
        if len(self.history_wct) == self.history_wct.maxlen:
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = self.history_samples[-1] - self.history_samples[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            samples_per_sec = elapsed_samples * world_size / elapsed_wct
            dev_samples_per_sec = elapsed_samples / elapsed_wct
            metrics.update(
                {
                    "throughput/batches_per_sec": elapsed_batches * world_size / elapsed_wct,
                    "throughput/samples_per_sec": samples_per_sec,
                    "throughput/device/batches_per_sec": elapsed_batches / elapsed_wct,
                    "throughput/device/samples_per_sec": dev_samples_per_sec,
                }
            )
            if lengths is not None:
                elapsed_lengths = int(self.history_lengths[-1]) - int(self.history_lengths[0])
                avg_length = elapsed_lengths / elapsed_batches  
                metrics.update(
                    {
                        "throughput/tokens_per_sec": samples_per_sec * avg_length,
                        "throughput/device/tokens_per_sec": dev_samples_per_sec * avg_length,
                        "total_tokens": avg_length * world_size * samples,
                    }
                )
                if train_loss is not None:
                    avg_loss = sum(self.history_training_loss) / len(self.history_training_loss)
                    metrics.update(
                        {
                            "metric/train_loss": avg_loss,
                            "metric/train_ppl": math.exp(avg_loss)
                        }
                    )

        if flops_per_batch is not None:
            self.history_flops.append(flops_per_batch * world_size)
        if len(self.history_flops) == self.history_flops.maxlen:
            elapsed_flops = sum(self.history_flops) - self.history_flops[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            flops_per_sec = elapsed_flops / elapsed_wct
            device_flops_per_sec = flops_per_sec / world_size
            metrics.update(
                {"throughput/flops_per_sec": flops_per_sec, "throughput/device/flops_per_sec": device_flops_per_sec}
            )
            if self.flops_available:
                metrics["throughput/device/mfu"] = device_flops_per_sec / self.flops_available

        metrics.update(
            {
                "time/train": train_elapsed / self.divider,
                "time/val": self.total_eval_wct / self.divider,
                "time/total": (train_elapsed + self.total_eval_wct) / self.divider,
                "samples": samples,
            }
        )
        if self.iter % self.log_iter_interval == 0:
            self.log_dict(metrics, step_count)

    def eval_end(self, eval_elapsed: float):
        self.total_eval_wct += eval_elapsed


class SpeedMonitorFabric(SpeedMonitorBase):
    def __init__(self, fabric: Fabric, *args: Any, **kwargs: Any) -> None:
        flops_available = get_flops_available(fabric.device, fabric._connector._precision_input)
        super().__init__(flops_available, fabric.log_dict, *args, **kwargs)

    @fabric_rank_zero_only
    def on_train_batch_end(self, *args: Any, **kwargs: Any):
        super().on_train_batch_end(*args, **kwargs)


class SpeedMonitorCallback(Callback):
    def __init__(self, length_fn: Callable[[Any], int], batch_size: int, **kwargs: Any) -> None:
        super().__init__()
        self.speed_monitor: Optional[SpeedMonitorBase] = None
        self.speed_monitor_kwargs = kwargs
        self.length_fn = length_fn
        self.batch_size = batch_size
        self.eval_t0: int = 0
        self.train_t0: int = 0
        self.total_lengths: int = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.speed_monitor is not None:
            return
        flops_available = get_flops_available(
            trainer.strategy.root_device, trainer._accelerator_connector._precision_flag
        )
        self.speed_monitor = SpeedMonitorBase(flops_available, trainer.logger.log_metrics, **self.speed_monitor_kwargs)

    @trainer_rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.fit_loop._should_accumulate():
            return

        self.train_t0 = time.perf_counter()

    @trainer_rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self.total_lengths += self.length_fn(batch)
        if trainer.fit_loop._should_accumulate():
            return
        train_elapsed = time.perf_counter() - self.train_t0
        assert self.speed_monitor is not None
        iter_num = trainer.fit_loop.total_batch_idx
        assert (measured_flops := pl_module.measured_flops) is not None
        self.speed_monitor.on_train_batch_end(
            (iter_num + 1) * self.batch_size,
            train_elapsed,
            trainer.world_size,
            flops_per_batch=measured_flops,
            lengths=self.total_lengths,
        )

    @trainer_rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_t0 = time.perf_counter()

    @trainer_rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        eval_elapsed = time.perf_counter() - self.eval_t0
        assert self.speed_monitor is not None
        self.speed_monitor.eval_end(eval_elapsed)