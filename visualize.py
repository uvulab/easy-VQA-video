import matplotlib.pyplot as plt

def plot_loss(history, filename='model_loss.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(filename)

def plot_accuracy(history, filename='model_accuracy.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Loss')
    plt.xlabel('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename)
