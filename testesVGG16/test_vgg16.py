import sys
sys.path.insert(0, '.')
from Modelos.VGG16 import build_vgg16, normalize_imagenet
import tensorflow as tf
import numpy as np

print('=== TESTE 1: Shape do modelo ===')
model = build_vgg16()
dummy = tf.random.normal((1, 32, 32, 3))
output = model(dummy)
print(f'Input shape: {dummy.shape}')
print(f'Output shape: {output.shape}')
assert output.shape == (1, 10), f'Esperado (1,10), obtido {output.shape}'
print('OK - Shape correto')

print('\n=== TESTE 2: Normalizacao ===')
img_test = np.random.randint(0, 256, (32, 32, 3), dtype='uint8')
img_norm = normalize_imagenet(img_test)
print(f'Valores originais: [{img_test.min()}, {img_test.max()}]')
print(f'Valores normalizados: [{img_norm.min():.2f}, {img_norm.max():.2f}]')
assert img_norm.min() >= -3 and img_norm.max() <= 3, 'Valores fora do range esperado'
print('OK - Normalizacao dentro do range')

print('\n=== TESTE 3: Contagem de parametros ===')
total_params = model.count_params()
print(f'Total de parametros: {total_params:,}')
print('OK - Modelo construido com sucesso')

print('\n=== TODOS OS TESTES PASSARAM ===')
