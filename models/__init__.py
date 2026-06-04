from .classical import run_classical_svm

from .qsvm import run_qsvm_train_test, run_qsvm_cv, run_qsvm_combined
from .qknn import run_qknn_train_test
from .qdt import run_qdt_train_test
from .qnn import run_qnn_train_test
from .qcnn import run_qcnn_train_test

# Generative (QGAN)
from .qgan import train_qgan_1d, sample_qgan_1d, QGANResult
