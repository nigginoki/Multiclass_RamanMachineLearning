import sys

sys.path.append("./lib/")

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from raman_lib.crossvalidation import CrossValidator
from raman_lib.misc import load_data, int_float, TqdmStreamHandler

# Prepare logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logdir = Path("./log/07_ANN_model/")

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
dt = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = logdir / f"{dt}.log"

handler_c = TqdmStreamHandler()
handler_f = logging.FileHandler(logfile)

handler_c.setLevel(logging.INFO)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_c.setFormatter(format_c)
handler_f.setFormatter(format_f)

logger.addHandler(handler_c)
logger.addHandler(handler_f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, action="store",
                        help="Path for the output directory.", required=True)
    parser.add_argument("-s", "--scoring", metavar="SCORE", type=str, nargs="*", action="store",
                        help="Scoring metrics to use. The first one will be used for hyperparameter selection.", default="accuracy")
    parser.add_argument("-t", "--trials", metavar="INT", type=int, action="store",
                        help="Number of trials for the randomized nested crossvalidation.", default=20)
    parser.add_argument("-k", "--folds", metavar="INT", type=int, action="store",
                        help="Number of folds for crossvalidation.", default=5)
    parser.add_argument("-j", "--jobs", metavar="INT", type=int, action="store",
                        help="Number of parallel jobs. Set as -1 to use all available processors", default=1)
    parser.add_argument("--epochs", metavar="INT", type=int, default=1000,
                        help="Number of epochs for training the ANN.")
    parser.add_argument("--batch-size", metavar="INT", type=int, default=32,
                        help="Batch size for training the ANN.")
    parser.add_argument("--learning-rate", metavar="FLOAT", type=float, default=0.0001,
                        help="Learning rate for the Adam optimizer.")
    
    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args

if __name__ == "__main__":

    logger.info("Classification using an artificial neural network")
    args = parse_args()

    path_in = Path(args.file)
    path_out = Path(args.out)

    filename = path_in.stem

    logger.info(f"Loading data from {path_in}")
    data = load_data(path_in)
   
    logger.info("Parsing data")
    X = data.drop(columns=["label", "file"], errors="ignore")
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = pd.factorize(data.label)[0]
    y_one_hot = label_binarize(y, classes=np.unique(y))
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    if isinstance(args.scoring, str):
        refit = True
    else:
        refit = args.scoring[0]
    
    # Define ANN model
    logger.info("Classification using an artificial neural network")
    ANN_path_out = path_out / "ANN"

    if not os.path.exists(ANN_path_out):
        logger.debug("Creating output directory")
        os.makedirs(ANN_path_out)

    # Define model
    model = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(y_one_hot.shape[1], activation='softmax') # Number of neurons equals number of classes
    ])
    
    # Compile model with a custom learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # loss function essential for classification task
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    logger.info("Starting training")
    
    # Set up early stopping callback, stopping around 200 epochs if there is no increase in model perfromance
    early_stopping = EarlyStopping(patience=200)
    
    # Train model (important batch_size)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=32,
        verbose=False,
        callbacks=[early_stopping])
    
    logger.info("Training complete")
    
    # Evaluate model performance
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    #cv.to_csv(tree_path_out)
    #export_graphviz(cv.estimator_, 
    #                out_file=str(ANN_path_out / "ANN.dot"), 
    #                feature_names=wns,  
    #                class_names=y_key,  
    #                filled=True, rounded=True,  
    #                special_characters=True,
    #                leaves_parallel=True)

    #logger.info("Cross validation complete ANN")


