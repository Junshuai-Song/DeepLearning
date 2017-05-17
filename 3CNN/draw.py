# Auther: Alan
import matplotlib.pyplot as plt

def draw_flower():
    accu = [0.617647, 0.608824, 0.591176, 0.55, 0.561765]
    accu = [0.570588, 0.585294, 0.579412, 0.535294, 0.520588]

    # plt.plot([accu[0]] * M, label="base learner")
    plt.margins(0.08)
    plt.plot(accu, label="my")
    names = [0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(range(len(names)), names, rotation=45)
    plt.xlabel("Keep dropout")
    plt.ylabel("Accuracy rate (%)")
    # plt.title("CNN LeNet 5 (28*28)")
    plt.title("CNN LeNet 5 (26*26)")
    plt.ylim(0.5, 0.7)
    plt.show()

def draw_cifar():
    acc = [ 0.3481 , 0.3493 , 0.3392 , 0.3325 , 0.3159]

    # plt.plot([accu[0]] * M, label="base learner")
    plt.margins(0.08)
    plt.plot(acc, label="my")
    names = [0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(range(len(names)), names, rotation=45)
    plt.xlabel("Keep dropout")
    plt.ylabel("Accuracy rate (%)")
    # plt.title("CNN LeNet 5 (28*28)")
    plt.title("CNN LeNet 5")
    plt.ylim(0.3, 0.4)
    plt.show()

if __name__=='__main__':
    draw_cifar()



