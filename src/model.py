import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMIL(nn.Module): 
    def __init__(self, num_classes=3):
        super(SAMIL, self).__init__()
        self.L = 500
        self.B = 250
        self.D = 128
        self.K = 1
        self.num_classes = num_classes
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(50, 100, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(100, 200, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(200 * 4 * 4, self.L),
            nn.ReLU(),
        )
        
        self.feature_extractor_part3 = nn.Sequential(
            
            nn.Linear(self.L, self.B),
            nn.ReLU(),
            nn.Linear(self.B, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes),
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 200 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL
        A_V = self.attention_V(H)  # NxK
        A_V = torch.transpose(A_V, 1, 0)  # KxN
        A_V = F.softmax(A_V, dim=1)  # softmax over N   
        H = self.feature_extractor_part3(H)
        A_U = self.attention_U(H)  # NxK
        A_U = torch.transpose(A_U, 1, 0)  # KxN
        A_U = F.softmax(A_U, dim=1)  # softmax over N
        A = torch.exp(torch.log(A_V) + torch.log(A_U)) #numerically more stable?
        A = A/torch.sum(A)
        M = torch.mm(A, H)  # KxL #M can be regarded as final representation of this bag
        out = self.classifier(M)
        
        return out, A_V

    def predict_proba(self, x):
        logits = self.forward(x)
        proba = F.softmax(logits, dim=0)
        return proba