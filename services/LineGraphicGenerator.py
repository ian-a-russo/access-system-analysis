import matplotlib.pyplot as plt

class LineGraphicGenerator:
    def __init__(self):
        pass

    def execute(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
