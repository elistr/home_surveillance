# FaceRecogniser.
# Brandon Joffe
# 2016
#
# Copyright 2016, Brandon Joffe, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code used in this project included opensource software (openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University

import argparse
import atexit
import logging
import math
import os
import os.path
import pickle
import threading
import time
from operator import itemgetter
from subprocess import Popen, PIPE

import aligndlib
import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import openface
from tensorflow_scripts.label_image import run_label

logger = logging.getLogger(__name__)

start = time.time()
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
luaDir = os.path.join(fileDir, '..', 'batch-represent')
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
args = parser.parse_args()

class FaceRecogniser(object):
    """This class implements face recognition using Openface's
    pretrained neural network and a Linear SVM classifier. Functions
    below allow a user to retrain the classifier and make predictions
    on detected faces"""

    def __init__(self, tf=True):
        self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.neuralNetLock = threading.Lock()
        self.predictor = dlib.shape_predictor(args.dlibFacePredictor)
        self.isTf = tf
        self.tf = FaceRecogniserNW()

        logger.info("Opening classifier.pkl to load existing known faces db")
        with open("generated-embeddings/classifier.pkl", 'r') as f: # le = labels, clf = classifier
            (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM

    def make_prediction(self,rgbFrame,bb, original_frame):
        """The function uses the location of a face
        to detect facial landmarks and perform an affine transform
        to align the eyes and nose to the correct positiion.
        The aligned face is passed through the neural net which
        generates 128 measurements which uniquly identify that face.
        These measurements are known as an embedding, and are used
        by the classifier to predict the identity of the person"""
        landmarks = self.align.findLandmarks(rgbFrame, bb)
        if landmarks == None:
            logger.info("///  FACE LANDMARKS COULD NOT BE FOUND  ///")
            return None
        alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            logger.info("///  FACE COULD NOT BE ALIGNED  ///")
            return None

        logger.info("////  FACE ALIGNED  // ")
        with self.neuralNetLock :
            if self.isTf:
                persondict = self.tf.recognize_face(alignedFace, original_frame)
                persondict['rep'] = self.getRep(alignedFace)
            else:
                persondict = self.recognize_face(alignedFace)

        if persondict is None:
            logger.info("/////  FACE COULD NOT BE RECOGNIZED  //")
            return persondict, alignedFace
        else:
            logger.info("/////  FACE RECOGNIZED  /// ")
            return persondict, alignedFace

    def recognize_face(self,img):
        if self.getRep(img) is None:
            return None
        rep1 = self.getRep(img) # Gets embedding representation of image
        logger.info("Embedding returned. Reshaping the image and flatting it out in a 1 dimension array.")
        rep = rep1.reshape(1, -1)   #take the image and  reshape the image array to a single line instead of 2 dimensionals
        start = time.time()
        logger.info("Submitting array for prediction.")
        predictions = self.clf.predict_proba(rep).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
        #logger.info("We need to dig here to know why the probability are not right.")
        maxI = np.argmax(predictions)
        person1 = self.le.inverse_transform(maxI)
        confidence1 = int(math.ceil(predictions[maxI]*100))

        logger.info("Recognition took {} seconds.".format(time.time() - start))
        logger.info("Recognized {} with {:.2f} confidence.".format(person1, confidence1))

        persondict = {'name': person1, 'confidence': confidence1, 'rep':rep1}
        return persondict

    def getRep(self,alignedFace):
        bgrImg = alignedFace
        if bgrImg is None:
            logger.error("unable to load image")
            return None

        logger.info("Tweaking the face color ")
        alignedFace = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        start = time.time()
        logger.info("Getting embedding for the face")
        rep = self.net.forward(alignedFace) # Gets embedding - 128 measurements
        return rep

    def reloadClassifier(self):
        with open("generated-embeddings/classifier.pkl", 'r') as f: # Reloads character stream from pickle file
            (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
        logger.info("reloadClassifier called")
        return True

    def trainClassifier(self):
        """Trainng the classifier begins by aligning any images in the
        training-images directory and putting them into the aligned images
        directory. Each of the aligned face images are passed through the
        neural net and the resultant embeddings along with their
        labels (names of the people) are used to train the classifier
        which is saved to a pickle file as a character stream"""

        logger.info("trainClassifier called")

        path = fileDir + "/aligned-images/cache.t7"
        try:
            os.remove(path) # Remove cache from aligned images folder
        except:
            logger.info("Failed to remove cache.t7. Could be that it did not existed in the first place.")
            pass

        start = time.time()
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",args.dlibFacePredictor,args.imgDim)
        logger.info("Aligning images for training took {} seconds.".format(time.time() - start))
        done = False
        start = time.time()

        done = self.generate_representation()

        if done is True:
            logger.info("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
            start = time.time()
            # Train Model
            self.train("generated-embeddings/","LinearSvm",-1)
            logger.info("Training took {} seconds.".format(time.time() - start))
        else:
            logger.info("Generate representation did not return True")


    def generate_representation(self):
        logger.info("lua Directory:    " + luaDir)
        self.cmd = ['/usr/bin/env', 'th', os.path.join(luaDir, 'main.lua'),'-outDir',  "generated-embeddings/" , '-data', "aligned-images/"]
        logger.info("lua command:    " + str(self.cmd))
        if args.cuda:
            self.cmd.append('-cuda')
            logger.info("using -cuda")
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
        #our issue is here, torch probably crashes without giving much explanation.
        outs, errs = self.p.communicate() # Wait for process to exit - wait for subprocess to finish writing to files: labels.csv & reps.csv
        logger.info("Waiting for process to exit to finish writing labels and reps.csv" + str(outs) + " - " + str(errs))

        def exitHandler():
            if self.p.poll() is None:
                logger.info("<=Something went Wrong===>")
                self.p.kill()
                return False
        atexit.register(exitHandler)

        return True


    def train(self,workDir,classifier,ldaDim):
        fname = "{}labels.csv".format(workDir) #labels of faces
        # logger.info("Loading labels " + fname + " csv size: " +  str(os.path.getsize("/root/home_surveillance/system/generated-embeddings/reps.csv")))
        if os.path.getsize(fname) > 0:
            logger.info(fname + " file is not empty")
            labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
            logger.info(labels)
        else:
            logger.info(fname + " file is empty")
            labels = "1:aligned-images/dummy/1.png"  #creating a dummy string to start the process
        logger.debug(map(os.path.dirname, labels))
        logger.debug(map(os.path.split,map(os.path.dirname, labels)))
        logger.debug(map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels))))
        labels = map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels)))

        fname = "{}reps.csv".format(workDir) # Representations of faces
        fnametest = format(workDir) + "reps.csv"
        logger.info("Loading embedding " + fname + " csv size: " + str(os.path.getsize(fname)))
        if os.path.getsize(fname) > 0:
            logger.info(fname + " file is not empty")
            embeddings = pd.read_csv(fname, header=None).as_matrix() # Get embeddings as a matrix from reps.csv
        else:
            logger.info(fname + " file is empty")
            embeddings = np.zeros((2,150)) #creating an empty array since csv is empty

        self.le = LabelEncoder().fit(labels) # LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1
        # Fits labels to model
        labelsNum = self.le.transform(labels)
        nClasses = len(self.le.classes_)
        logger.info("Training for {} classes.".format(nClasses))

        if classifier == 'LinearSvm':
            self.clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier == 'GMM':
            self.clf = GMM(n_components=nClasses)

        if ldaDim > 0:
            clf_final =  self.clf
            self.clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                ('clf', clf_final)])

        self.clf.fit(embeddings, labelsNum) #link embeddings to labels

        fName = "{}/classifier.pkl".format(workDir)
        logger.info("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((self.le,  self.clf), f) # Creates character stream and writes to file to use for recognition

    def getSquaredl2Distance(self,rep1,rep2):
        """Returns number between 0-4, Openface calculated the mean between
        similar faces is 0.99 i.e. returns less than 0.99 if reps both belong
        to the same person"""

        d = rep1 - rep2


class FaceRecogniserNW(object):
    """This class implements face recognition using googles tenserflow and imagenet pretrained models"""
    def __init__(self):
        pass

    def recognize_face(self, face_img, original_frame):
        cv2.imwrite('/host/hacks/face_img.jpeg', face_img)
        cv2.imwrite('/host/hacks/original_frame.jpeg', original_frame)
        fname = "/host/hacks/face_img.jpeg"

        predictions = run_label(fname)

        return predictions[0]

    def run_recognize_on_windows(self, original_numpy_image_array):

        # and each window will be 220x220 image size
        # Accutal possible area is a square: (400,400) ~ (1700, 1100)
        IMAGE_X = 640
        IMAGE_Y = 480
        SIZE = 224

        X1 = 0
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        counter = 0
        file_names_list = []
        while X1 <= (IMAGE_X - SIZE):
            Y1 = 0
            while Y1 <= (IMAGE_Y - SIZE):
                # import pdb; pdb.set_trace()
                crop_img = original_numpy_image_array[Y1:Y1 + SIZE, X1:X1 + SIZE]
                # resized = cv2.resize(crop_img, (220, 299), interpolation=cv2.INTER_AREA)
                file_name = "temp/temp_%d.jpeg" % counter
                counter += 1

                cv2.imwrite(file_name, crop_img)
                file_names_list.append(file_name)

                Y1 += int(SIZE / 2)
            X1 += int(SIZE / 4)
        print('Created %d windows images.' % counter)

        predictions = run_inference_on_multiple_images(file_names_list)
        print('There are %d predictions' % len(predictions.keys()))
