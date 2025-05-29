import matplotlib.pyplot as plt
import seaborn as sns


def show_batch(batch, nrows=4, ncols=4, figsize=2):
    assert len(batch['target']) == ncols * nrows
    images = batch['image'].numpy().transpose(0, 2, 3, 1)
    figure, axs = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))

    for n, (img, label) in enumerate(zip(images, batch['target'])):
        i, j = n // ncols, n % ncols
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].set_title(f'class - {label}')
        axs[i, j].set_axis_off()

    plt.show()


def plot_confusion_matrix(cm, title, class_labels=None):
    plt.clf(), plt.cla()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    return plt.gcf()
