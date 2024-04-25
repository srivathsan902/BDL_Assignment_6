import io
import sys
import numpy as np
import keras
import argparse
from PIL import Image
import keras.models as models
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File