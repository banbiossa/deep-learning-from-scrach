# coding: utf-8
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _numerical_gradient_1d(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        for idx in range(x.size):
                tmp_val = x[idx]
                x[idx] = float(tmp_val) + h
                fxh1 = f(x)  # f(x+h)

                x[idx] = tmp_val - h
                fxh2 = f(x)  # f(x-h)
                grad[idx] = (fxh1 - fxh2) / (2*h)

                x[idx] = tmp_val  # 値を元に戻す

        return grad


def numerical_gradient_2d(f, X):
        if X.ndim == 1:
                return _numerical_gradient_1d(f, X)
        else:
                grad = np.zeros_like(X)

                for idx, x in enumerate(X):
                        grad[idx] = _numerical_gradient_1d(f, x)

                return grad


def numerical_gradient(f, x):
        ''' Calculates numerical gradient on a point.
        The point is a single vector in a multi dimentional space.

        Parameters
        ----------
        f: function
                takes x as input for function
        x: np.ndarray
                input of f. 
                the gradient will be calculated on all elements.

        Returns
        -------
        grad: gradient of f, shape of x. 
        '''
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
                idx = it.multi_index
                tmp_val = x[idx]
                x[idx] = float(tmp_val) + h
                fxh1 = f(x)  # f(x+h)

                x[idx] = float(tmp_val) - h
                fxh2 = f(x)  # f(x-h)
                grad[idx] = (fxh1 - fxh2) / (2*h)

                x[idx] = tmp_val  # 値を元に戻す
                it.iternext()

        return grad


def numerical_gradient_on_array(f, x_array):
        ''' Returns gradient on multiple points.

        Parameters
        ----------
        f: functions
                function that takes elements of x_array as input
        x_array: np.ndarray
                elements will be inputs of f

        Returns
        -------
        grad: np.ndarray
                gradinet, shape of x_array
        '''
        logger.info(f"Calculate gradient on {f.__name__}")
        logger.info(f"Shape of input is {x_array.shape}")

        grad = np.zeros_like(x_array)
        for idx, x in enumerate(x_array):
                logger.debug(f"idx: {idx}, x: {x}")
                grad[idx] = numerical_gradient(f, x)

        return grad
