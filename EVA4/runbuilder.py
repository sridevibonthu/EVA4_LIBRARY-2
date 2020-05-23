# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # TensorBoard support

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
import os

# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
from collections import namedtuple
from itertools import product

# Read in the hyper-parameters and return a Run namedtuple containing all the 
# combinations of hyper-parameters
class RunBuilder():
  @staticmethod
  def get_runs(params):

    Run = namedtuple('Run', params.keys())

    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))
    
    return runs

# Helper class, help track loss, accuracy, epoch time, run time, 
# hyper-parameters etc. Also record to TensorBoard and write into csv, json
class RunManager():
  def __init__(self, savepath, channel_means, channel_stdevs, classification=True, visdecoder={}):

    # tracking every epoch count, loss, accuracy, time
    self.epoch_count = 0
    self.batch_count = 0
    self.min_val_loss = 10e10
    self.epoch_train_loss = 0
    self.classification = classification
    self.epoch_train_num_correct = 0
    self.epoch_start_time = None
    self.batch_start_time = None
    self.savepath = savepath

    # tracking every run count, run data, hyper-params used, time
    self.run_params = None
    self.run_count = 0
    self.run_data = []
    self.run_start_time = None
    self.visdecoder = visdecoder # how to decode the outcome for visualization


    # record model, trainloader and TensorBoard 
    self.network = None
    self.trainloader = None
    self.tb = None
    self.channel_means = channel_means
    self.channel_stdevs = channel_stdevs

  def sample_outcome_images(self, outcomes, suffix='input'):
      l = 0
      for k, v in self.visdecoder.items():
          vgrid = torchvision.utils.make_grid(outcomes[:32,l:v,:,:])
          l = v
          self.tb.add_image(f'{k}({suffix})', vgrid)

  def sample_outcome_classes(self, outcomes, suffix='input'):
      l = 0
      for k, v in self.visdecoder.items():
          if v-l == 1:
            vgrid = torchvision.utils.make_grid(outcomes[:32,l:v,:,:])
          else:
            # for each pixel, set the value to index of argmax. if max is do nothing
            # add all zeros tensot at beginning
            
            o = outcomes[:32,l:v,:,:]
            s = o.size()
            o = torch.cat((o, torch.zeros(s[0], 1, s[2], s[3])))
            o = (o.argmax(dim=1)+1)%(v-l+1)
            vgrid = torchvision.utils.make_grid(o)
            # test by creating random tensor and visualize
          l = v
          self.tb.add_image(f'{k}({suffix})', vgrid)
          

  # record the count, hyper-param, model, trainloader of each run
  # record sample images and network graph to TensorBoard  
  def begin_run(self, run, network, trainloader, testloader):

    self.run_start_time = time.time()

    self.run_params = run
    self.run_count += 1

    self.network = network
    self.trainloader = trainloader
    self.testloader = testloader

    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(self.savepath, 
        f'runs/{current_time}_{socket.gethostname()}-{run}')

    self.tb = SummaryWriter(log_dir)

    images, outcomes = next(iter(self.trainloader))
    grid = torchvision.utils.make_grid(images[:32])
    for i in range(grid.size()[0]):
        grid[i] = grid[i] * self.channel_stdevs[i] + self.channel_means[i]

    self.tb.add_image('images', grid)
    self.sample_outcome_images(outcomes)
    
    self.tb.add_graph(self.network, images.to(self.network.device))

  # when run ends, close TensorBoard, zero epoch count
  def end_run(self):
    self.tb.close()
    self.epoch_count = 0
    self.batch_count = 0

  def begin_batch(self):
      self.batch_start_time = time.time()

  def end_batch(self, lr):
      self.batch_count += 1
      self.tb.add_scalar('Batch Learning Rate', lr, self.batch_count)
      batch_duration = time.time() - self.batch_start_time
      return batch_duration

  # zero epoch count, loss, accuracy, 
  def begin_epoch(self):
    self.epoch_start_time = time.time()
    self.epoch_count += 1
    self.epoch_train_loss = 0
    self.epoch_train_num_correct = 0
    self.epoch_test_loss = 0
    self.epoch_test_num_correct = 0
    self.test_output = None
    self.test_input = None

  def end_epoch(self, lr):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    # record epoch loss and accuracy
    trainloss = self.get_train_loss()
    testloss = self.get_test_loss()
    self.tb.add_scalar('Train Loss', trainloss, self.epoch_count)
    self.tb.add_scalar('Test Loss', testloss, self.epoch_count)
    self.tb.add_scalar('Learning Rate', lr, self.epoch_count)

    # output sample images created in test
    self.sample_outcome_classes(self.test_output, f'output-{self.epoch_count}')
    self.sample_outcome_images(self.test_input, f'input-{self.epoch_count}')

    results = OrderedDict()
    results["run"] = self.run_count
    results["epoch"] = self.epoch_count
    results["train loss"] = trainloss
    results["test loss"] = testloss

    if self.classification:
        train_accuracy = self.get_train_accuracy()
        test_accuracy = self.get_test_accuracy()
        self.tb.add_scalar('Train Accuracy', train_accuracy, self.epoch_count)
        self.tb.add_scalar('Test Accuracy', test_accuracy, self.epoch_count)
        results["train accuracy"] = train_accuracy
        results["test accuracy"] = test_accuracy

    # Record params to TensorBoard
    for name, param in self.network.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
    
    # Write into 'results' (OrderedDict) for all run related data
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration   
    
    # Record hyper-params into 'results'
    for k,v in self.run_params.items(): 
        results[k] = v
    self.run_data.append(results)
    
    return f'{results}'

  # accumulate loss of batch into entire epoch loss
  def track_train_loss(self, loss):
    # multiply batch size so variety of batch sizes can be compared
    self.epoch_train_loss += loss.item() * self.trainloader.batch_size

  def track_test_loss(self, loss):
    self.epoch_test_loss += loss.item() * self.testloader.batch_size

  # accumulate number of corrects of batch into entire epoch num_correct
  def track_train_num_correct(self, preds, labels):
    if self.classification:
        self.epoch_train_num_correct += self._get_num_correct(preds, labels)

  def track_test_num_correct(self, preds, labels):
    self.test_input = labels
    self.test_output = preds
    if self.classification:
        self.epoch_test_num_correct += self._get_num_correct(preds, labels)

  def get_train_accuracy(self):
      return self.epoch_train_num_correct / len(self.trainloader.dataset)

  def get_test_accuracy(self):
      return self.epoch_test_num_correct / len(self.testloader.dataset)

  def get_test_loss(self):
      return self.epoch_test_loss / len(self.testloader.dataset)

  def get_train_loss(self):
      return self.epoch_train_loss / len(self.trainloader.dataset)

  @torch.no_grad()
  def _get_num_correct(self, preds, labels):
    # this is for label encoding. For onehot this must change
    # we need to find 
    #return preds.argmax(dim=1).eq(labels).sum().item()
    return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
  
  def savebest(self, fileName):
      f = os.path.join(self.savepath, f'{fileName}.pt')
      if self.epoch_test_loss < self.min_val_loss:
          torch.save(self.network.state_dict(), f)

  # save end results of all runs into csv, json for further analysis
  def save(self, fileName):
    f = os.path.join(self.savepath, f'{fileName}.csv')
    pd.DataFrame.from_dict(
        self.run_data, 
        orient = 'columns',
    ).to_csv(f)

    


