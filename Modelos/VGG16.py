import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from wandb.integration.keras import WandbMetricsLogger
import wandb


def get_cifar10_labels():
    builder = tfds.builder("cifar10")
    builder.download_and_prepare()
    return builder.info.features['label'].names


def normalize_imagenet(x):
    # Normaliza com estatisticas ImageNet
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    x = x.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    return x


def build_vgg16():
    # Arquitetura VGG16 para CIFAR-10 (32x32x3)
    # Conv blocks: 64,64->128,128->256,256,256->512,512,512->512,512,512
    # FC: 512->4096->4096->10

    model = Sequential([
        # Block 1: 64, 64
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 2: 128, 128
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 3: 256, 256, 256
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 4: 512, 512, 512
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 5: 512, 512, 512
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Flatten para FC layers
        Flatten(),

        # FC layers: 4096, 4096, 10 (SEM dropout conforme tasks)
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model


def trainer_vgg16(config):
    print(f"--- Iniciando treinamento VGG16 from scratch ---")

    try:
        class_names = get_cifar10_labels()
    except Exception as e:
        print(f"Erro ao carregar labels via TFDS: {e}")
        return

    # Carrega CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Guarda copia RAW para plotar depois
    x_test_raw = x_test.copy()

    # Normaliza com ImageNet stats
    x_train = normalize_imagenet(x_train)
    x_test = normalize_imagenet(x_test)

    # Encoding das classes
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Constroi modelo VGG16
    model = build_vgg16()

    # Compila com SGD+momentum
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=config['learning_rate'],
            momentum=0.9
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_list = [
        WandbMetricsLogger(log_freq='epoch')
    ]

    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )

    # Matriz de Confusao
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Matriz interativa WandB
    try:
        wandb.log({
            "conf-matrix_interactive": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_classes,
                preds=y_pred_classes,
                class_names=class_names,
                title='Matriz de Confusão - CIFAR-10',
            )
        })
    except Exception as e:
        print(f"Aviso: Nao foi possivel gerar matriz nativa do WandB: {e}")

    # Matriz estatica
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.xticks(rotation=45)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão - CIFAR-10')
    plt.tight_layout()

    wandb.log({"conf_mat_image": wandb.Image(fig)})
    plt.close(fig)

    # Exemplo de validacao
    idx = random.randint(0, len(x_test) - 1)
    sample_image = x_test_raw[idx]

    real_class = y_true_classes[idx]
    predicted_class = y_pred_classes[idx]

    real_class_name = class_names[real_class]
    predicted_class_name = class_names[predicted_class]

    print(f"Enviando exemplo de validacao")
    wandb.log({
        "validaton.exemple": wandb.Image(
            sample_image,
            caption=f"index: {idx} | Real: {real_class_name} | Predicted: {predicted_class_name}"
        )
    })

    final_loss = history.history['val_loss'][-1]
    final_accuracy = history.history['val_accuracy'][-1]

    print(f"--- Treino VGG16 finalizado ---")
    print(f"Final loss: {final_loss}")
    print(f"Final accuracy: {final_accuracy}")

    return {"loss": final_loss, "accuracy": final_accuracy}


def create_augmented_versions(image, n_versions=5):
    """
    Cria versões aumentadas de uma imagem usando data augmentation

    Transformações aplicadas (tf.keras.layers):
    - RandomFlip (horizontal)
    - RandomRotation (15 graus)
    - RandomTranslation (width/height)
    - RandomZoom

    Args:
        image: Imagem original (32, 32, 3) com valores 0-255
        n_versions: Número de versões aumentadas para gerar

    Returns:
        Lista de imagens aumentadas
    """
    # Cria pipeline de augmentation usando camadas modernas
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.042),  # 15 graus em radianos (~15/360)
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    augmented_images = []

    # Normaliza para [0,1]
    img_normalized = image.astype('float32') / 255.0
    # Adiciona dimensão batch
    img_batch = tf.expand_dims(img_normalized, 0)

    # Gera n versões aumentadas
    for i in range(n_versions):
        # Aplica augmentation
        img_aug = augmentation(img_batch, training=True)[0]
        # Clipa valores para [0,1]
        img_aug = tf.clip_by_value(img_aug, 0, 1)
        # Converte de volta para [0, 255]
        img_aug = (img_aug.numpy() * 255).astype('uint8')

        augmented_images.append(img_aug)

    return augmented_images


def plot_augmented_versions(original_image, augmented_images, save_path='augmentation_test.png'):
    """
    Plota a imagem original e suas versões aumentadas

    Args:
        original_image: Imagem original
        augmented_images: Lista de imagens aumentadas
        save_path: Caminho para salvar a figura
    """
    n_images = len(augmented_images) + 1  # +1 para a original

    fig, axes = plt.subplots(1, n_images, figsize=(3*n_images, 3))

    # Plota original
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Plota versões aumentadas
    for i, img_aug in enumerate(augmented_images):
        axes[i+1].imshow(img_aug)
        axes[i+1].set_title(f'Augmented {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figura salva em {save_path}")
    plt.close()


# Teste de data augmentation (quando rodar o arquivo diretamente)
if __name__ == "__main__":
    print("Testando data augmentation")
    print("="*60)

    # Carrega CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(f"Dataset carregado - Train: {len(x_train)}, Test: {len(x_test)}")

    # Pega uma imagem aleatória
    idx = np.random.randint(0, len(x_train))
    sample_image = x_train[idx]
    print(f"Imagem selecionada: indice {idx}")

    # Gera 5 versões aumentadas
    print("Gerando 5 versoes aumentadas")
    augmented = create_augmented_versions(sample_image, n_versions=5)
    print(f"Geradas {len(augmented)} versoes")

    # Plota e salva
    print("Plotando versoes")
    plot_augmented_versions(sample_image, augmented)
    print("Teste de augmentation concluido")
