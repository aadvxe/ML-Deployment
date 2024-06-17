from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.models import load_model
import pickle
from collections import defaultdict

app = FastAPI()


