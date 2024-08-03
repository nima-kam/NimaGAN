import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import transforms

class NoiseDataset(Dataset):

  def __init__(self,file_name,start_column=1,latent_size=512):
    noise_df=pd.read_csv(file_name).iloc[:,start_column:]
    self.latent_size=latent_size
    self.init_params(df=noise_df)


  def init_params(self,df):
    self.noise_df=df
    self.x=self.noise_df.iloc[:,0:self.latent_size].values

    self.y=self.noise_df.iloc[:].label.values


  def __len__(self):
    return len(self.x)

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]


class NoiseClassifier(nn.Module):
    """
    NoiseClassifier uses a classifier to generate labels for images produced by the generator given a latent code.

    Attributes:
        classifier : The classifier model used to generate labels.
        generator : The generator model used to synthesize images.
        transform : A function/transform that takes in a tensor and returns a transformed version.
        latent_space_type (str): The type of latent space ('Z' or 'W').
        device (str): The device on which the computation will take place ('cpu' or 'cuda').
    """

    def __init__(self, classifier, generator, transform=None, latent_space_type='Z', device='cpu'):
        super(NoiseClassifier, self).__init__()
        self.cl = classifier
        self.gen = generator
        self.latent_sp = latent_space_type
        self.device = device
        if transform is not None:
          self.transform = transform
        else:
            self.transforms = transforms.Compose([
                              transforms.Resize((256, 256)),
                              # transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
                              ])
        if (self.gen.gan_type in ['stylegan', 'stylegan2']) and self.latent_sp == 'W':
            self.synthesis_kwargs = {'latent_space_type': 'W'}
        else:
            self.synthesis_kwargs = {}
        print(f'GAN type set as {self.synthesis_kwargs}')

    def forward(self, code):
        images = self.gen.easy_synthesize(code, **self.synthesis_kwargs)['image']
        moved = np.moveaxis(images, -1, 1)
        images = torch.tensor(moved, dtype=torch.float32, device=self.device)

        if self.transform:
            sample = self.transform(images)
        else:
            sample = images

        labels = self.cl(sample)
        return labels




class LatentDataset(Dataset):
    """
    A dataset for handling latent codes and their corresponding scores.

    Attributes:
        latents (np.array): Array of latent codes.
        scores (np.array): Array of scores corresponding to the latent codes.
    """

    def __init__(self, latents, scores):
        self.x = latents
        self.s = scores

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.s[idx]
    
    
    


class NNBoundary(nn.Module):
    """
    NNBoundary defines a neural network with configurable hidden layers and neurons 
    to perform classification on latent codes directly. The network gradient for the 
    input latent will be used as the editing direction.

    Attributes:
        input_size (int): The size of the input layer.
        num_of_hidden_layers (int): The number of hidden layers.
        num_of_hidden_neuron (int): The number of neurons in each hidden layer.
        act_func (callable): The activation function for hidden layers.
        device (str): The device on which the computation will take place ('cpu' or 'cuda').
        name (str): the neural network name for saving.
    """

    def __init__(self, input_size=512, num_of_hidden_layers=1, num_of_hidden_neuron=256, act_func=nn.ReLU, device='cpu',name=""):
        super(NNBoundary, self).__init__()
        self.input_size = input_size
        self.hidden_l = num_of_hidden_layers
        self.hidden_n = num_of_hidden_neuron
        self.act_func = act_func
        self.device = device
        self.name=name

        self.input = nn.Parameter(torch.randn(input_size, dtype=torch.float64), requires_grad=True).to(device)
        self.flatten = nn.Flatten()

        layer_stack = []
        for i in range(self.hidden_l):
            if i == 0:  # first hidden layer
                layer_stack.append(nn.Linear(self.input_size, self.hidden_n, dtype=torch.float64))
                layer_stack.append(self.act_func())
                layer_stack.append(nn.Dropout(0.1))
            else:
                layer_stack.append(nn.Linear(self.hidden_n, self.hidden_n, dtype=torch.float64))
                layer_stack.append(self.act_func())

        if self.hidden_l > 0:
            layer_stack.append(nn.Linear(self.hidden_n, 1, dtype=torch.float64))
        else:  # no hidden layer
            layer_stack.append(nn.Linear(self.input_size, 1, dtype=torch.float64))

        self.linear_elu_stack = nn.Sequential(*layer_stack).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.flatten(x)  # flatten Wp vectors

        logits = self.linear_elu_stack(x)
        return logits
    
    def set_name(self, name):
        self.name = name
    def save_pkl(self, base_path='late_models/'):
        name= f'{self.name}_{self.hidden_l}_{self.hidden_n}.pkl'        
        torch.save(self,os.path.join(base_path, name))
    
class PolyBoundary:
    """
    PolyBoundary trains a neural network classifier to find a boundary for latent codes.

    Attributes:
        X (np.array): Array of latent codes.
        Y (np.array): Array of scores corresponding to the latent codes.
        out_path (str): Path to save the trained Network.        
        split_ratio (float): Ratio to split the training and validation sets.
        chosen_num_or_ratio (float): Number or ratio of samples chosen for training.
        invalid_value (float): Value to filter invalid samples.
    """

    def __init__(self, X, Y, split_ratio=0.7, chosen_num_or_ratio=0.8, invalid_value=None):
        self.x = X  # shape: (n_samples, n_features)
        self.boundary_shape = (1, X.shape[1])
        self.y = Y  # shape: (n_samples,)
        self.invalid_value = invalid_value
        self.split_ratio = split_ratio
        self.chosen_num_or_ratio = chosen_num_or_ratio

    def train_nn(self, nn_classifier, epochs=10, lr=0.01, invalid_value=None):
        split_ratio = self.split_ratio
        chosen_num_or_ratio = self.chosen_num_or_ratio
        print(f'Filtering training data.')
        scores = self.y
        latent_codes = self.x

        if invalid_value is not None:
            latent_codes = latent_codes[scores[:, 0] != invalid_value]
            scores = scores[scores[:, 0] != invalid_value]

        print(f'Sorting scores to get positive and negative samples.')
        sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
        latent_codes = latent_codes[sorted_idx]
        scores = scores[sorted_idx]
        num_samples = latent_codes.shape[0]

        if 0 < chosen_num_or_ratio <= 1:
            chosen_num = int(num_samples * chosen_num_or_ratio)
        else:
            chosen_num = int(chosen_num_or_ratio)
        chosen_num = min(chosen_num, num_samples // 2)

        print(f'Splitting training and validation sets:')
        train_num = int(chosen_num * split_ratio)
        val_num = chosen_num - train_num

        # Positive samples
        positive_idx = np.arange(chosen_num)
        np.random.shuffle(positive_idx)
        positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
        positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]

        # Negative samples
        negative_idx = np.arange(chosen_num)
        np.random.shuffle(negative_idx)
        negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
        negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]

        # Training set
        train_data = np.concatenate([positive_train, negative_train], axis=0)
        train_label = np.concatenate([np.ones(train_num, dtype=np.int32),
                                      np.zeros(train_num, dtype=np.int32)], axis=0)
        print(f'\n  Training: {train_num} positive, {train_num} negative.')

        # Validation set
        val_data = np.concatenate([positive_val, negative_val], axis=0)
        val_label = np.concatenate([np.ones(val_num, dtype=np.int32),
                                    np.zeros(val_num, dtype=np.int32)], axis=0)
        print(f'\n  Validation: {val_num} positive, {val_num} negative.')

        # Remaining set
        remaining_num = num_samples - chosen_num * 2
        remaining_data = latent_codes[chosen_num:-chosen_num]
        remaining_scores = scores[chosen_num:-chosen_num]
        decision_value = (scores[0] + scores[-1]) / 2
        remaining_label = np.ones(remaining_num, dtype=np.int32)
        remaining_label[remaining_scores.ravel() < decision_value] = 0
        remaining_positive_num = np.sum(remaining_label == 1)
        remaining_negative_num = np.sum(remaining_label == 0)
        print(f'\n  Remaining: {remaining_positive_num} positive, {remaining_negative_num} negative.')

        train_ds = LatentDataset(train_data, train_label)
        train_dl = DataLoader(train_ds, batch_size=42, shuffle=True)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adamax(nn_classifier.parameters(), lr=lr, weight_decay=5e-4)

        print(f'Training NN.')
        nn_classifier.train()

        for e in range(epochs):  # epochs
            running_loss = 0.
            running_corrects = 0.

            for i, (inputs, labels) in tqdm(enumerate(train_dl), mininterval=4.):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).type(torch.float).unsqueeze(1)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = nn_classifier(inputs.type(torch.float64)).to(self.device).type(torch.float32)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()
                running_loss += loss.item()

            last_loss = running_loss / len(train_dl)  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))

        print(f'Finish training.')

        nn_classifier.eval()
        val_ds = LatentDataset(val_data, val_label)
        val_dl = DataLoader(val_ds, batch_size=26, shuffle=False)

        rem_ds = LatentDataset(remaining_data, remaining_label)
        rem_dl = DataLoader(rem_ds, batch_size=26, shuffle=False)

        def eval_model(classifier, dl, loss_fn, name='train'):
            classifier.eval()
            with torch.no_grad():
                val_accs = 0.0
                val_loss = 0.0
                for images, labels in tqdm(dl):
                    images = images.to(self.device)
                    labels = labels.to(self.device).type(torch.float).unsqueeze(1)
                    outputs = classifier(images).type(torch.float)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    # Calculate accuracies for each label
                    predicted_labels = torch.sigmoid(outputs) > 0.5
                    val_accs += accuracy_score(labels.detach().cpu().numpy(), predicted_labels.cpu())
                print(f'Accuracy for {name} set: {val_accs} / {len(dl)} , {val_accs/len(dl)*100:.3f}%,\nLoss for {name} set: {val_loss / len(dl)}')

        eval_model(nn_classifier, train_dl, loss_fn, name='train')

        if val_num:
            eval_model(nn_classifier, val_dl, loss_fn, name='validation')

        if remaining_num:
            eval_model(nn_classifier, rem_dl, loss_fn, name='remaining')


def cal_latent_derivative(start_late, late_classifier,device='cpu'):
    """
    Classify the latent and calculate the derivative of latent code.

    Args:
        start_late (np.array): The starting latent code.
        late_classifier (nn.Module): The latent classifier.

    Returns:
        np.array: The calculated derivative.
    """
    crit = nn.BCELoss()
    l2 = torch.Tensor(start_late).to(device).type(torch.float64)
    l2 = torch.tensor(l2, requires_grad=True)

    late_classifier.eval()
    late_classifier.zero_grad()
    a = torch.sigmoid(late_classifier(l2))
    a.backward()

    a_bound = l2.grad.clone().cpu().numpy()
    a_bound = (a_bound) / np.linalg.norm(a_bound)  # normalize the generated grad
    return a_bound

def apply_manipulation(start_late, late_classifier, multiplier=1, steps=1, lin_boundary=None):
    """
    Apply manipulation based on the latent classifier gradient to the latent code.

    Args:
        start_late (np.array): The starting latent code.
        late_classifier (nn.Module): The latent classifier.
        multiplier (float): The maximum multiplier for the manipulation.
        steps (int): The number of steps for the manipulation.
        lin_boundary (np.array): The linear boundary. 

    Returns:
        np.array: The manipulated latent code.
    """
    m = 1 if multiplier > 0 else -1
    step_m = abs(multiplier) / max(steps, 1)
    ptemp_late = start_late.copy()

    for i in range(max(int(steps), 1)):
        b = cal_latent_derivative(ptemp_late, late_classifier)
        ptemp_late += b * step_m * m

    if lin_boundary is not None:
        lp_late = start_late.copy() + multiplier * lin_boundary
        return ptemp_late, lp_late

    return ptemp_late