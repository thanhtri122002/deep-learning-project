e inputs to the appropriate format
    train_images = np.array(train_images)
    train_boxes = [np.array(image_boxes) for image_boxes in train_boxes]
    train_labels = [np.array(image_labels) for image_labels in train_labels]
    val_images = np.array(val_images)
    val_boxes = [np.array(image_boxes) for image_boxes in val_boxes]
    val_labels = [np.array(image_labels) for image_labels in val_labels]

    return train_images, train_boxes, train_labels, val_images, val_boxes, val_labels


def define_backbone_alexnet(dimension):
    input = Input(shape = dimension)

    x = Conv2D(filters=96, kernel_size=11, strides=4, name='conv1', activation='relu')(input)
    x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)
    x =