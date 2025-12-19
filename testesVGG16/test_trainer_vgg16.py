import sys
sys.path.insert(0, '.')
from Modelos.VGG16 import trainer_vgg16
import wandb
import os
from dotenv import load_dotenv

# Carrega variaveis de ambiente
load_dotenv()

# Faz login no WandB usando API key do .env
wandb_key = os.getenv('WANDB_KEY')
if wandb_key:
    wandb.login(key=wandb_key)
else:
    print("AVISO: WANDB_KEY nao encontrada no .env")

# Inicializa WandB
wandb.init(
    project="VC-Trabalho-Pratico",
    name="test-vgg16-1epoch",
    config={
        "architecture": "vgg16",
        "dataset": "CIFAR-10",
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.01
    }
)

config = {
    "architecture": "vgg16",
    "dataset": "CIFAR-10",
    "epochs": 1,
    "batch_size": 128,
    "learning_rate": 0.01
}

print("=== TESTE TRAINER VGG16 (1 epoca) ===")
metrics = trainer_vgg16(config)

print("\n=== RESULTADO ===")
print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")

wandb.finish()
print("\n=== TESTE COMPLETO ===")
