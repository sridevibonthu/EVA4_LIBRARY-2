import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import copy
from tqdm.notebook import trange, tqdm

class LRRangeFinder():
  def __init__(self, model, optimizer, criterion, epochs, start_lr, end_lr, dataloader):
    self.model = model
    self.optimizer = optimizer
    self.epochs = epochs
    self.criterion = criterion
    self.start_lr = start_lr
    self.end_lr = end_lr
    self.dataloader = dataloader
    self.modelstate = copy.deepcopy(self.model.state_dict())
    self.optimstate = copy.deepcopy(self.optimizer.state_dict())

    
  def range_test(self):
    iter = 0
    smoothing = 0.05
    self.loss = []
    self.lr = []

    #criterion = nn.CrossEntropyLoss() 
    lr_lambda = lambda x: math.exp(x * math.log(self.end_lr / self.start_lr) / (self.epochs * len(self.dataloader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    for i in trange(self.epochs):
      for inputs, labels in tqdm(self.dataloader):
        
        # Send to device
        inputs = inputs.to(self.model.device)
        labels = labels.to(self.model.device)
        
        # Training mode and zero gradients
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get outputs to calc loss
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update LR
        scheduler.step()
        lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.lr.append(lr_step)

        # smooth the loss
        if iter==0:
          self.loss.append(loss)
        else:
          loss = smoothing  * loss + (1 - smoothing) * self.loss[-1]
          self.loss.append(loss)
        
        iter += 1
        #print(iter, end="*")
      
    plt.ylabel("loss")
    plt.xlabel("Learning Rate")
    plt.xscale("log")
    plt.plot(self.lr, self.loss)
    plt.show()

    self.model.load_state_dict(self.modelstate)
    self.optimizer.load_state_dict(self.optimstate)

    return(self.lr[self.loss.index(min(self.loss))])