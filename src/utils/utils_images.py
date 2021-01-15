import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

to_pil = torchvision.transforms.ToPILImage()


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''

    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def matplotlib_imshow(img, one_channel=False):
    """
    helper function to show an image
    (used in the `plot_classes_preds` function below)

    :param img:
    :param one_channel:
    :return:
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 6))
    n = len(images)
    if len(images) > 16:
        n = 16

    for idx in np.arange(n):
        ax = fig.add_subplot(1, n, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))

    return fig


def plot_images_with_distances(images, dist_an, dist_ap):
    batch_size = len(images[0])
    if len(images[0]) > 16:
        batch_size = 16

    fig, ax_arr = plt.subplots(batch_size, 3, figsize=(16, 3 * batch_size))
    for idx, row in enumerate(ax_arr):

        for i, ax in enumerate(row):
            img = to_pil(images[i][idx])
            if i == 0:
                ax.set_title(f'Anchor')
            else:
                match = dist_ap[idx] < dist_an[idx]
                if i == 1:
                    ax_title = f'Positive Match Anchor: {match}\ndist_ap: {dist_ap[idx].item()}'
                if i == 2:
                    ax_title = f'Negative Match Anchor: {not match}\ndist_an: {dist_an[idx].item()}'

                ax.set_title(ax_title, color="green" if match else "red")
            ax.imshow(img)

    fig.suptitle('Displaying random triplets with their predicted distances', fontsize=16)

    return fig


def plot_classes_preds_face_recognition(images, labels, predictions, fuzzy_matches=True):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels

    fig = plt.figure(figsize=(2.5 * len(images), 3))
    n = len(images)
    if n > 16:
        n = 16

    for idx in np.arange(n):
        ax = fig.add_subplot(1, n, idx + 1, xticks=[], yticks=[])

        img = to_pil(images[idx])
        if fuzzy_matches:
            if int(predictions[idx][0][0]) == int(labels[idx]):
                match_dist = predictions[idx][0][1]
                title = f"True match:\ndist: {match_dist}\n(label: {labels[idx]})"
                color = "green"
            else:
                fuzzy_match = False
                for (i, match) in enumerate(predictions[idx]):
                    if int(labels[idx]) == int(match[0]):
                        fuzzy_match = True
                        fuzz_match_idx = i
                        match_dist = match[1]
                if fuzzy_match:
                    title = f"fuzzy match:\npos: {fuzz_match_idx}, " \
                            f"dist: {match_dist}\n(label: {labels[idx]})"
                    color = "orange"
                else:
                    title = f"Wrong prediction:\n(label: {labels[idx]})"
                    color = "red"
        else:
            title = f"prediction: {predictions[idx]}\n(label: {labels[idx]})"
            color = "green" if int(labels[idx]) == int(predictions[idx]) else "red"

        ax.imshow(img)
        ax.set_title(title, color=color)

    fig.suptitle('Displaying random positive images with their predictions', fontsize=16)

    return fig