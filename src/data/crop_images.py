from PIL import Image
from glob import glob


def crop_into_16(train_path, label_path, index):
    # Open the image file
    img = Image.open(train_path)
    label = Image.open(label_path)

    # Loop over the 16 sub-images
    for i in range(16):
        # Calculate the coordinates for the sub-image
        left = max(0, i % 4 * 128 - i % 4 - 1)
        upper = max(0, i // 4 * 128 - i // 4 - 1)
        right = left + 132
        lower = upper + 132

        # Crop the sub-image from the original image
        sub_image = img.crop((left, upper, right, lower))
        sub_label = label.crop((left, upper, right, lower))

        # Save the sub-image to a file
        sub_image.save(f"data/raw/132x132/train_images/train_{index}_{i + 1}.png")
        sub_label.save(f"data/raw/132x132/train_labels/labels_{index}_{i + 1}.png")


train_images = glob("data/raw/508x508/train_images/*")
train_labels = glob("data/raw/508x508/train_labels/*")

for j, (train_img, train_label) in enumerate(zip(train_images, train_labels)):
    crop_into_16(train_img, train_label, j)
