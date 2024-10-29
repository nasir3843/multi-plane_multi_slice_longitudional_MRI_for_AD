#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from keras.layers import Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from tensorflow.keras.layers import Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, multiply, Conv2D, Add, Activation, Concatenate
import os
import argparse


def get_backbone_model(backbone_name, input_shape):
    if backbone_name == "EfficientNet":
        return applications.EfficientNetB7(include_top=False, input_shape=input_shape)
    elif backbone_name == "XceptionNet":
        return applications.Xception(include_top=False, input_shape=input_shape)
    elif backbone_name == "ResNet":
        return applications.ResNet50(include_top=False, input_shape=input_shape)
    elif backbone_name == "ConvNext":
        return applications.ConvNeXtBase(include_top=False, input_shape=input_shape)
    elif backbone_name == "DenseNet":
        return applications.DenseNet121(include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone model {backbone_name}")

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)
    
    avg_pool = tf.keras.layers.Dense(channel // ratio, activation='relu')(avg_pool)
    avg_pool = tf.keras.layers.Dense(channel, activation='sigmoid')(avg_pool)
    
    max_pool = tf.keras.layers.Dense(channel // ratio, activation='relu')(max_pool)
    max_pool = tf.keras.layers.Dense(channel, activation='sigmoid')(max_pool)
    
    scale = Add()([avg_pool, max_pool])
    scale = Activation('sigmoid')(scale)
    return multiply([input_feature, scale])

# Spatial Attention Module
def spatial_attention(input_feature):
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return multiply([input_feature, attention])

# CBAM Block
def cbam_block(input_feature):
    x = channel_attention(input_feature)
    x = spatial_attention(x)
    return x

# Define classification heads
def combine_backbone_head(backbone, input_shape, head_name, cbam):
    if head_name == "mlp":
        inputs = layers.Input(shape=input_shape)
        x = backbone(inputs, training=False)        
        out=0
        if cbam:
            x = cbam_block(x)
            x = Flatten()(x)
            x = tf.keras.layers.Dense(128, activation='relu') (x)
            out = tf.keras.layers.Dense(1, activation='sigmoid') (x)
        else:
            x = GlobalAveragePooling2D()(x)
            # x = tf.expand_dims(x, axis=1)        
            x = tf.keras.layers.Dense(128, activation='relu') (x)
            out = tf.keras.layers.Dense(1, activation='sigmoid') (x)        

        cnn_mlp = tf.keras.Model(inputs=inputs, outputs=out)
        return cnn_mlp

    elif head_name == "lstm":        
        inputs = layers.Input(shape=input_shape)
        x = backbone(inputs, training=False)        
        out=0
        if cbam:
            x = cbam_block(x)
            x = Flatten()(x)
            x = tf.expand_dims(x, axis=1)
            x = tf.keras.layers.LSTM(128, activation='relu', dropout=0.3, return_sequences=True) (x)
            x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=False) (x)
            x = tf.keras.layers.Dense(128, activation='relu') (x)
            out = tf.keras.layers.Dense(1, activation='sigmoid') (x)
        else:
            x = GlobalAveragePooling2D()(x)
            x = tf.expand_dims(x, axis=1) 
            x = tf.keras.layers.LSTM(128, activation='relu', dropout=0.3, return_sequences=True) (x)
            x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=False) (x)
            x = tf.keras.layers.Dense(128, activation='relu') (x)
            out = tf.keras.layers.Dense(1, activation='sigmoid') (x)        
     
        cnn_lstm = tf.keras.Model(inputs=inputs, outputs=out)    
        return cnn_lstm

    elif head_name == "multihead_attention": 
        inputs = layers.Input(shape=input_shape)
        x = backbone(inputs, training=False)        
        out=0
        if cbam:
            x = cbam_block(x)
            x = Flatten()(x)
            x = tf.expand_dims(x, axis=1)
             # Define Multi-Head Attention layer with 8 heads and key dimension of 32
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
            # Layer Normalization and Dropout for regularization
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output)        
            x = tf.keras.layers.Dropout(0.3)(x)        
            # Flatten the output of attention layer for classification
            x = tf.squeeze(x, axis=1)        
            # Add a fully connected layer for binary classification
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid') (x)
            cnn_mha = tf.keras.Model(inputs=inputs, outputs=x)
        else:
            x = GlobalAveragePooling2D()(x)
            # x = Flatten()(x)
            x = tf.expand_dims(x, axis=1)
             # Define Multi-Head Attention layer with 8 heads and key dimension of 32
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
            # Layer Normalization and Dropout for regularization
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output)        
            x = tf.keras.layers.Dropout(0.3)(x)        
            # Flatten the output of attention layer for classification
            x = tf.squeeze(x, axis=1)        
            # Add a fully connected layer for binary classification
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid') (x)          
            cnn_mha = tf.keras.Model(inputs=inputs, outputs=x)        
        return cnn_mha
    else:
        raise ValueError(f"Unknown classification head {head_name}")

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    # y_pred = np.argmax(y_pred_prob, axis=1)
    # y_true = np.argmax(y_true, axis=1)    
    accuracy = accuracy_score(y_true, y_pred)    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    # auc = roc_auc_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    # print(f"AUC: {auc}")
    
    return accuracy, sensitivity, specificity
    
# Data loading and preprocessing
def load_data(data_dir, image_size=(121, 121)):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, )
    train_data = datagen.flow_from_directory(os.path.join(data_dir, 'train'), target_size=image_size, batch_size=2)
    valid_data = datagen.flow_from_directory(os.path.join(data_dir, 'valid'), target_size=image_size, batch_size=2) 
    test_data = datagen.flow_from_directory(os.path.join(data_dir, 'test'), target_size=image_size, batch_size=2)
    return train_data, valid_data, test_data

# Training and evaluating the model
def train_model(backbone, head, data_dir, epochs=2, cbam=False, image_size=(121, 121)):

    input_shape = (image_size[0], image_size[1], 3)
    model = get_backbone_model(backbone, input_shape)
    hybrid_model = combine_backbone_head(model, input_shape, head, cbam)
    hybrid_model.summary()
    hybrid_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Load data
    train_data, valid_data, test_data = load_data(data_dir, image_size=image_size)
    # Train the model
    hybrid_model.fit(train_data, validation_data=valid_data, epochs=epochs)

    # Evaluate the model on the test set
    y_true = test_data.labels
    y_pred_prob = hybrid_model.predict(test_data)
    y_out = y_pred_prob > 0.5 
    y_pred_int = y_out.astype(float)
    # Calculate metrics
    calculate_metrics(y_true, y_pred_int)


# backbone = 'EfficientNet'
# backbone_name = 'ResNet'
backbone_name = 'ConvNext'
# backbone_name = 'DenseNet'
# backbone_name = 'XceptionNet'

# head = 'lstm'
head = 'mlp'
# head = 'multihead_attention'

# cbam=True
cbam=False

data_dir = "D:\Code_Tutorial\VoMLab\Data"
# train_model(backbone_name, head, data_dir, cbam)


# Main function to parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hybrid CNN model with custom backbone and head.')
    parser.add_argument('--cnnbb', type=str, help='CNN backbone to use, e.g., EfficientNet, ResNet, XceptionNet, etc.')
    parser.add_argument('--ch', type=str, help='Classification head to use, e.g., mlp, lstm, multihead_attention')
    parser.add_argument('--cbam', type=bool, default=False, help='True or Fale for usign CBAM attention')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    args = parser.parse_args()

    train_model(args.cnnbb, args.ch, args.data_dir, epochs=args.epochs, cbam=args.cbam)

