def read_images(file):
    with open(file, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        images = []
        for _ in range(num_images):
            image = []
            for _ in range(rows * cols):
                pixel = int.from_bytes(f.read(1), "big")
                image.append(pixel)
            images.append(image)
    return images


def read_labels(file):
    with open(file, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        num_items = int.from_bytes(f.read(4), "big")
        labels = []
        for _ in range(num_items):
            label = int.from_bytes(f.read(1), "big")
            labels.append(label)
    return labels
