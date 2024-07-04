import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

def PGD_attack(image_path, epsilon=0.1, eps_step=0.0009, max_iter=300, targeted=False,target_label=0, decay=None, num_random_init=0, batch_size=32, random_eps=False, summary_writer=False, verbose=True):
    # Load pre-trained ResNet18 model
    model = resnet18(pretrained=True)

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    original_image = Image.open(image_path).convert('RGB')
    input_image = preprocess(original_image).unsqueeze(0)

    # Create PyTorch classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )

    # Create PGD attack instance with additional parameters
    pgd_attack = ProjectedGradientDescent(
        classifier,
        norm=np.inf,
        eps=epsilon,
        eps_step=eps_step,
        decay=decay,
        max_iter=max_iter,
        targeted=targeted,
        num_random_init=num_random_init,
        batch_size=batch_size,
        random_eps=random_eps,
        summary_writer=summary_writer,
        verbose=verbose,
    )
    if targeted:
        adv_image = pgd_attack.generate(x=input_image.numpy(), y=torch.tensor([target_label]).numpy())
    else:
        adv_image = pgd_attack.generate(x=input_image.numpy())


    # 在原图和对抗样本上进行预测
    pred_original = classifier.predict(input_image.numpy())
    pred_adversarial = classifier.predict(adv_image)


    adv_image_pil = Image.fromarray((adv_image[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    adv_image_pil = adv_image_pil.resize(original_image.size)

    # # 显示原始图像和对抗样本
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image)
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(adv_image_pil)
    # plt.title("Adversarial Image")
    # plt.axis('off')
    # plt.show()

    # adv_image_pil.save("adversarial_image.png")

    advimg_directory = "./advimg"
    os.makedirs(advimg_directory, exist_ok=True)
    adversarial_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_PGD{os.path.splitext(image_path)[1]}"
    adv_image_pil.save(os.path.join(advimg_directory, adversarial_filename))


    def get_class(i:int):
        labels_path = "imagenet-simple-labels.json"  # Replace with the actual path
        with open(labels_path, "r") as file:
            imagenet_labels = json.load(file)
        print(f"Class {i}: {imagenet_labels[i]}")
        return f"Class {i}: {imagenet_labels[i]}"

    # 获取原始图像的最终预测类别
    original_class = np.argmax(pred_original, axis=1)
    print("Original Predicted Class:", original_class.item())
    get_class(original_class.item())

    # 获取对抗图像的最终预测类别
    adversarial_class = np.argmax(pred_adversarial, axis=1)
    print("Adversarial Predicted Class:", adversarial_class.item())
    get_class(adversarial_class.item())

    re = "Original Predicted Class:", original_class.item(), '\n',get_class(original_class.item()),'\n',\
            "Adversarial Predicted Class:", adversarial_class.item(), '\n', get_class(adversarial_class.item())
    return re

if __name__ == "__main__":
    # Specify the image path and attack parameters
    image_path = "cat.png"

    PGD_attack(image_path,targeted=False,target_label=0)