import tqdm
import torch

class test:
    def __init__(self, model, loss_fn, objective_metric, device):
        self.sum = 0
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.objective_metric = objective_metric

    def computer_avg_score(self, test_set):
        for i in tqdm.tqdm(range(len(test_set))):
            img = test_set[i][0]
            label = test_set[i][1]

            image_tensor = torch.unsqueeze(img, 0)
            image_tensor = image_tensor.to(self.device)

            label_tensor = torch.unsqueeze(label, 0)
            label_tensor = label_tensor.to(self.device)

            pred = self.model(image_tensor)
            loss = self.loss_fn(pred, label_tensor)
            score = self.objective_metric(pred, label_tensor)

            self.sum += float(score.item())

        return sum / len(test_set)
