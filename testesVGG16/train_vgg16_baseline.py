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

# Inicializa WandB com config baseline
wandb.init(
    project="VC-Trabalho-Pratico",
    name="vgg16-baseline-20epochs",
    config={
        "architecture": "vgg16",
        "dataset": "CIFAR-10",
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "optimizer": "SGD",
        "momentum": 0.9
    }
)

# Configuracao baseline conforme tasks.txt
config = {
    "architecture": "vgg16",
    "dataset": "CIFAR-10",
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.01
}

print("=== TREINAMENTO BASELINE VGG16 ===")
print(f"Configuracao: {config}")
print("Iniciando treinamento completo com 20 epocas")

metrics = trainer_vgg16(config)

print("\n=== RESULTADO FINAL ===")
print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")

wandb.finish()
print("\n=== TREINAMENTO COMPLETO ===")
print("Resultados salvos no WandB")
