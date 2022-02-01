import cv2
import numpy as np
import torch
import pickle
import random

class Inference():
  def __init__(self, model, device, scale_size=(224,224)):
    self.scale_size = scale_size
    self.model = model
    self.device = device
  def predict(self,img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image, self.scale_size) / 255.
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_resize.transpose(2, 0, 1))).float()
    image_tensor = torch.unsqueeze(image_tensor,0)
    image_tensor = image_tensor.to(self.device)
    output = model(image_tensor)
    prediction = output[0].cpu().detach().numpy()
    if prediction.shape[0] == 2:
      prediction = np.argmax(prediction, axis=0)
    else:
      indices_0 = prediction < 0.5
      indices_1 = prediction >= 0.5
      prediction[indices_0] = 0
      prediction[indices_1]=1
    return output, prediction
