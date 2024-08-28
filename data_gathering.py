from sklearn.datasets import load_iris


def get_data():
  iris = load_iris()
  return iris.data, iris.target