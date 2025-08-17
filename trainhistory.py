import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

filename="training_history.csv"
history = pd.read_csv(filename)

# 7. 可视化训练过程 (从第5个epoch开始)
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'][:], label='train loss')
plt.plot(history['val_loss'][:], label='val loss')
plt.title('from epoch 5: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'][:], label='training accuracy')
plt.plot(history['val_acc'][:], label='validation accuracy')
plt.title('from epoch 5: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()