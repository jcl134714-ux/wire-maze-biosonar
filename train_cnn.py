"""
CNN classifier for wire-maze echo spectrograms.
Network architecture matches Table 1 in the paper.

Usage:
    python train_cnn.py --data_dir data/ --epochs 50 --batch_size 64
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


CLASS_NAMES = [
    'SquareLattice', 'PoissonDisk', 'HexagonalLattice',
    'InhomRandom', 'Cluster',
]


def build_model(input_shape=(121, 49, 3), n_classes=5):
    """
    Build CNN classifier (Table 1).

    4 conv blocks: 32 -> 64 -> 128 -> 256 channels.
    Blocks 1-3: Conv2D + BatchNorm + MaxPool2D(2x2).
    Block 4:    Conv2D + BatchNorm + GlobalAveragePooling2D.
    Head:       Dense(512, 256, 128, 64) with dropout(0.5, 0.4, 0.3, 0.3).
    Output:     Dense(5, softmax).
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    # Block 4: 256 filters + GlobalAvgPool
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='WireMaze_CNN')


def train(data_dir, epochs, batch_size):
    # Load datasets
    train_npz = np.load(f'{data_dir}/train_data.npz')
    val_npz = np.load(f'{data_dir}/val_data.npz')
    test_npz = np.load(f'{data_dir}/test_data.npz')

    X_train = train_npz['spectrograms']
    y_train = np.argmax(train_npz['labels'], axis=1)
    X_val = val_npz['spectrograms']
    y_val = np.argmax(val_npz['labels'], axis=1)
    X_test = test_npz['spectrograms']
    y_test = np.argmax(test_npz['labels'], axis=1)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build and compile
    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=4000,
        decay_rate=0.9,
        staircase=True
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
    )

    # Evaluate on test set
    _, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Confusion matrix on test set
    y_pred = np.argmax(model.predict(X_test, batch_size=batch_size), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, f"Test Accuracy: {test_acc:.4f}")

    # Save
    model.save('trained_model.keras')
    print("Model saved to trained_model.keras")
    return model, history


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(9, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues', ax=ax, im_kw={'aspect': 'equal'})
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size)
