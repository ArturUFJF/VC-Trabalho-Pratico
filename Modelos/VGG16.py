import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
