
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

# --- Model Definition (Must match exactly) ---
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {"val_loss": loss.detach(), "val_accuracy": acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_accuracy"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_accuracy": epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_accuracy']))

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def test_inference():
    print("Loading model...")
    model = ResNet9(3, 38)
    try:
        model.load_state_dict(torch.load('plant_disease_weights.pth', map_location='cpu'))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    print("Model loaded.")

    # Create a dummy image (e.g., random noise or a solid color)
    # Ideally we'd use a real image, but random noise simulates "something"
    # to maintain determinism, let's create a gradient or specific pattern if possible, 
    # but uniform random is fine to check logit variance.
    # Actually, let's create 3 different fake images.
    
    test_images = [
        torch.rand(1, 3, 256, 256), # Random noise
        torch.ones(1, 3, 256, 256) * 0.5, # Grey
        torch.zeros(1, 3, 256, 256) # Black
    ]
    
    print("\n--- Testing with NO Normalization (0-1 range) ---")
    for i, img_tensor in enumerate(test_images):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            print(f"Image {i}: Pred Class {pred.item()}, Conf {conf.item():.4f}")
            # print("  Top 5 logits:", output[0].topk(5).values.tolist())
            # print("  Top 5 indices:", output[0].topk(5).indices.tolist())

    print("\n--- Testing with Normalization (Standard ImageNet) ---")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    for i, img_tensor in enumerate(test_images):
        # Apply normalization manually
        # img_tensor is [1, 3, H, W], normalize expects [3, H, W]
        img_norm = normalize(img_tensor.squeeze(0)).unsqueeze(0)
        with torch.no_grad():
            output = model(img_norm)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            print(f"Image {i}: Pred Class {pred.item()}, Conf {conf.item():.4f}")

if __name__ == "__main__":
    test_inference()
