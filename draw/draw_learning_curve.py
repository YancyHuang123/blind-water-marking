import matplotlib
import pickle
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.pyplot import MultipleLocator # type: ignore


def draw_learning_curve(folder,file_name):
    losses = None
    with open(f'{folder}/{file_name}', 'rb') as handle:
        losses = pickle.load(handle)

    for name,v in losses.items():
        plt.plot(range(1, len(v[0])+1), v[0],label=f'{name}') 


    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2)) 

    plt.xlabel('epoch') #设置坐标标注
    plt.ylabel('loss')


    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title("losses by epoch") # 设置title

    plt.savefig(f'{folder}/learning_curve.jpg')