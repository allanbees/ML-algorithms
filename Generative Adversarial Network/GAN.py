import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

class Classifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.001),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.001),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.002),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(32),
            nn.Linear(32, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)

# Params: digit of MNIST dataset and batch size
def load_data(digit, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )

    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    if digit >= 0 and digit < 10:
        idx = dataset.train_labels==digit
        dataset.targets = dataset.train_labels[idx]
        dataset.data = dataset.train_data[idx] 
        
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True ,shuffle=True) 
    return loader

def train_classifier(opt, model, x_true, x_false, accuracy=None, max_iters=100, batch_size=1000):
    lr_G = 3e-4
    noise_dim = 64
    img_dim = 28 * 28 * 1
    generator = Generator(noise_dim, img_dim).to(device)
    opt_gen = optim.Adam(generator.parameters(), lr=lr_G)
    criterion = nn.BCELoss() 
    real_positive = 0
    fake_positive = 0
    acc_C = 0
    for it in range(max_iters):
        for batch_idx, (real, _) in enumerate(x_true):
            real = real.view(-1, 784).to(device)
            real_positive = 0
            fake_positive = 0
            # Train classifier
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake = generator(noise)
            clas_real = model(real).view(-1)
            lossC_real = criterion(clas_real, torch.ones_like(clas_real))
            clas_fake = model(fake.detach()).view(-1)
            lossC_fake = criterion(clas_fake, torch.zeros_like(clas_fake))
            lossC = (lossC_real + lossC_fake) / 2
            model.zero_grad()
            lossC.backward(retain_graph=True)
            opt.step()
            
            
            if batch_idx == 0:
                print(
                    f"Epoch [{it}/{max_iters}] \
                          Loss Classifier: {lossC:.4f}"
                )
                    
                if accuracy != None:
                    for i in range(clas_real.shape[0]):
                        if clas_real[i] > 0.95:
                            real_positive += 1
                    
                    for i in range(clas_fake.shape[0]):
                        if clas_fake[i] < 0.001:
                            fake_positive += 1
                            
                    acc_C = (real_positive+fake_positive)/(clas_fake.shape[0]+clas_real.shape[0])
                    print(f"Classificator accuracy is: {acc_C}")    
                    if acc_C >= accuracy:
                        print("Accuracy reached...")
                        return
                    
                fake, acc_G = train_generator(opt_gen, generator, model, criterion, None, 200)
                if acc_G == 1:
                    return
                with torch.no_grad():
                    fake = generator(x_false).reshape(-1,1,28,28)
                    fake_example  = fake[5] 
                    img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)
                    save_image(fake_example, f"fakes_produced/epoch_{it}.png")
                    save_image(img_grid_fake, f"fakes_produced_grid/epoch_{it}.png")
        
                
def train_generator(opt, generator, classifier, criterion, accuracy=None, max_iters=30, batch_size=1000):
    acc_G = 0 
    threshold = 0.5
    if accuracy != None:
        for it in range(max_iters):
            fake_noise = torch.randn(batch_size, 64).to(device)
            fake = generator(fake_noise)
            output = classifier(fake).view(-1)
            loss = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
        total = 0
        for i in range(batch_size):
            if output[i] > threshold:
                total += 1
        acc_G = total / batch_size
        print(f"Generator accuracy is: {acc_G/batch_size}")
        if acc_G >= accuracy:
            print("Generator accuracy reached...")
            acc_G = 1
        
    else:
        for it in range(max_iters):
            fake_noise = torch.randn(batch_size, 64).to(device)
            fake = generator(fake_noise)
            output = classifier(fake).view(-1)
            loss = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

    return fake, acc_G

    
def GAN(epochs=50): 
    lr_C = 3e-4
    img_dim = 28 * 28 * 1
    batch_size = 64
    noise_dim = 64

    # load the data
    loader = load_data(4, batch_size)
    
    # Classifier and generator instances
    classifier = Classifier(img_dim).to(device)
    opt_clas = optim.Adam(classifier.parameters(), lr=lr_C)
    fixed_noise = torch.randn((batch_size, noise_dim)).to(device)
    train_classifier(opt_clas, classifier, loader, fixed_noise, None, epochs)

GAN()