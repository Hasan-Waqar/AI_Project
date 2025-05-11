import numpy as np
y_cls_train = np.load("y_cls_train.npy")  # One-hot encoded Gear_output
gear_labels = np.argmax(y_cls_train, axis=1)  # Convert one-hot to indices (0 to 7)
gear_values = [-1, 0, 1, 2, 3, 4, 5, 6]
gear_counts = {gear_values[i]: np.sum(gear_labels == i) for i in range(8)}
print("Gear distribution in training data:", gear_counts)