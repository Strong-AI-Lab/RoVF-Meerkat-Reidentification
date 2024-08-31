import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, decay_steps: int, start_lr: float, max_lr: float, end_lr: float, last_epoch=-1):
        self.warmup_steps = warmup_steps
        assert warmup_steps > 0, f"Warmup steps must be greater than 0. Got {warmup_steps}"
        self.decay_steps = decay_steps
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.end_lr = end_lr
        self.current_step = 0
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def update_current_step(self, current_step): # calling this takes place of old last_epoch
        # current_step should be the number of batch updates so far. Used when reinitializing a model.
        self.current_step = current_step 

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.current_step / self.warmup_steps)
        elif self.current_step < self.decay_steps:
            # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)))
            lr = self.end_lr + (self.max_lr - self.end_lr) * cosine_decay
        else:
            # Constant learning rate
            lr = self.end_lr
        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            self.current_step += 1
        else:
            self.current_step = epoch
        super(WarmupCosineDecayScheduler, self).step(epoch)

if __name__ == "__main__":
    
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)

    # Parameters for the scheduler
    warmup_steps = 10
    decay_steps = 100
    start_lr = 1e-5
    max_lr = 1e-3
    end_lr = 1e-4

    scheduler = WarmupCosineDecayScheduler(optimizer, warmup_steps, decay_steps, start_lr, max_lr, end_lr)

    # Simulate stepping through the scheduler and record LR values
    lr_values = []
    for step in range(decay_steps + 20):  # Go beyond the decay steps to see the flat end_lr
        scheduler.step()
        lr_values.append(scheduler.get_lr()[0])

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, label='Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Cosine Decay Scheduler')
    plt.legend()
    plt.grid(True)
    plt.show()

    #plt.savefig('pathhere.png')