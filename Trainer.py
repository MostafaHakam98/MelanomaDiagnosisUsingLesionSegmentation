import abc
import os
import sys
import tqdm
import torch
import datetime
from math import cos,pi
import numpy as np
import pickle
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Callable, Any
from typing import NamedTuple, List
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from pynvml import *


def to_np(x):
    return x.data.cpu().numpy()

class BatchResult(NamedTuple):
    loss: float
    score: float


class EpochResult(NamedTuple):
    losses: List[float]
    score: float


class FitResult(NamedTuple):
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    best_score: float

class Trainer:
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 objective_metric,
                 device,
                 learning_rate = None,
                 callbacks=[],
                 tensorboard_logger=None,
                 tensorboard_log_images=True,
                 experiment_prefix=None
                 ):

        self.tensorboard_logger = tensorboard_logger

        if experiment_prefix is None:
            now = datetime.datetime.now()
            self.experiment_prefix = now.strftime("%Y-%m-%d\%H:%M:%S")
        else:
            self.experiment_prefix = experiment_prefix
        self.tensorboard_log_images = tensorboard_log_images
        self.model = model
        self.loss_fn = loss_fn
        nvmlInit()
        self.device_info=nvmlDeviceGetHandleByIndex(0)
        self.optimizer = optimizer
        self.lr = learning_rate
        self.objective_metric = objective_metric
        self.device = device
        self.callbacks=callbacks
        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            empty_cache=False,
            early_stopping: int = None,
            best_score=None,
            current_epoch=0,
            print_every=1, **kw) -> FitResult:

        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        epochs_without_improvement = 0

        for epoch in range(current_epoch,num_epochs):
            if empty_cache:
              torch.cuda.empty_cache()

            for callback in self.callbacks:
              if callback=='exponential_decay_lr':
                self.adjust_learning_rate(epoch=epoch, max_epoch=num_epochs)
              if callback=='cosine_decay':
                self.adjust_learning_rate(step=epoch, decay_steps=num_epochs)
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            epoch_train_res = self.train_epoch(dl_train, verbose=verbose, **kw)
            
            train_loss.extend([float(x.item()) for x in epoch_train_res.losses])
            train_acc.append(float(epoch_train_res.score))

            epoch_test_res = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.extend([float(x.item()) for x in epoch_test_res.losses])
            test_acc.append(float(epoch_test_res.score))

            if best_score is None:
                best_score = epoch_test_res.score
            elif epoch_test_res.score > best_score:
                best_score = epoch_test_res.score
                if checkpoints is not None:
                    if not os.path.exists(checkpoints):
                        os.makedirs(checkpoints)

                    if not os.path.exists(checkpoints + 'best_models/'):
                        os.makedirs(checkpoints + 'best_models/')

                    torch.save(self.model, checkpoints + 'best_models/'+self.model.__class__.__name__.split('_')[0]+'_'+str(best_score)+'_'+str(epoch))
                    print("**** New Best Weights Saved ****")
                epochs_without_improvement = 0

            else:
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    print("Early stopping after %s with out improvement" % epochs_without_improvement)
                    break
                if checkpoints is not None:
                    if not os.path.exists(checkpoints):
                        os.makedirs(checkpoints)

                    if not os.path.exists(checkpoints + 'intermediate_models/'):
                        os.makedirs(checkpoints + 'intermediate_models/')

                    print("**** Intermediate Weights Saved ****")
                    torch.save(self.model, checkpoints+'intermediate_models/'+self.model.__class__.__name__.split('_')[0]+'_'+str(epoch)+'_'+str(epoch_test_res.score))
                epochs_without_improvement += 1

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc, best_score)

    def adjust_learning_rate(self, epoch, max_epoch, power=0.9):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = round(self.lr * np.power(1-(epoch) / max_epoch, power), 8)
    
    def cosine_decay(self,step,decay_steps):
      alpha=1e-7
      cosine_decay = 0.5 * (1 + cos(pi * epoch / max_epochs))
      decayed = (1 - alpha) * cosine_decay + alpha
      for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr*decayed
    
    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        self.model.train()  
        return self._foreach_batch(self.device_info,dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(self.device_info,dl_test, self.test_batch, **kw)

    def train_batch(self, index, batch_data) -> BatchResult:
        X, y = batch_data
        ############TO BE CONSIDERED##############
        if self.tensorboard_logger and self.tensorboard_log_images:
            B = torch.zeros_like(X.squeeze())
            C = torch.stack([B, X.squeeze(), X.squeeze()])
            C = C.unsqueeze(dim=0)
            images = C
            grid = make_grid(images, normalize=True, scale_each=True)
            self.tensorboard_logger.add_image("exp-%s/batch/test/images" % self.experiment_prefix, grid, index)
        ###############END######################
        if isinstance(X, tuple) or isinstance(X, list):
            X = [x.to(self.device) for x in X]
        else:
            X = X.to(self.device)
        y = y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        score = self.objective_metric(pred, y)
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/loss' % self.experiment_prefix, loss, index)
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/score' % self.experiment_prefix, score, index)            
        return BatchResult(loss, score)

    def test_batch(self, index, batch_data) -> BatchResult:
        with torch.no_grad():
            X, y = batch_data
            if isinstance(X, tuple) or isinstance(X, list):
                X = [x.to(self.device) for x in X]
            else:
                X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)

            loss = self.loss_fn(pred, y)
            
            score = self.objective_metric(pred, y)
            
            if self.tensorboard_logger:
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/loss' % self.experiment_prefix, loss, index)
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/score' % self.experiment_prefix, score, index)
            return BatchResult(loss, score)

    @staticmethod
    def _print(message, verbose=True):
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(device_info,dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        losses = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            overall_score = overall_loss = avg_score = avg_loss = counter = 0
            min_loss = min_score = 1
            max_loss = max_score = 0
            for batch_idx in range(num_batches):
                counter += 1
                data = next(dl_iter)
                batch_res = forward_fn(batch_idx, data)
                
                if batch_res.loss > max_loss:
                    max_loss = batch_res.loss
                if batch_res.score > max_score:
                    max_score = batch_res.score

                if batch_res.loss < min_loss:
                    min_loss = batch_res.loss
                if batch_res.score < min_score:
                    min_score = batch_res.score
                overall_loss += batch_res.loss
                overall_score += batch_res.score
                losses.append(batch_res.loss)
                info = nvmlDeviceGetMemoryInfo(device_info)

                avg_loss = overall_loss / counter
                avg_score = overall_score / counter
                gpu_usage = (info.used / info.total) * 100
                pbar.set_description(f'{pbar_name} (Avg. loss:{avg_loss:.3f}, Avg. score:{avg_score:.3f}, Used GPU Memory: {gpu_usage:.3f}%)')
                pbar.update()

            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, Min {min_loss:.3f}, Max {max_loss:.3f}), '
                                 f'(Avg. Score {avg_score:.4f}, Min {min_score:.4f}, Max {max_score:.4f})')

        return EpochResult(losses=losses, score=avg_score)
