from datasets import load_dataset
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import hashlib
import io
import re
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm

train = load_dataset(
    "Multimodal-Fatima/CUB_train",
    split="train"
)

test = load_dataset(
    "Multimodal-Fatima/CUB_test",
    split="test"
)

print("Training samples:", len(train))
print("Testing samples:", len(test))
#Columns: ['image', 'description', 'label', 'file_name', 'id']...
print("Columns:", train.column_names)
print("First record:")
print(train[0])


image = train[0]["image"]

print("Image size:", image.size)
print("Image width:", image.width)
print("Image height:", image.height)
print("Image mode:", image.mode)


sizes = Counter(
    example["image"].size
    for example in train
)

print("Number of different image sizes:", len(sizes))
print("\nMost common image sizes:")

for size, count in sizes.most_common(10):
    print(f"{size}: {count} images")

#(500, 333): 897 images
#(500, 375): 595 images
#(500, 400): 175 images
#(500, 334): 173 images
#(333, 500): 159 images
#(500, 357): 122 images
#(500, 332): 115 images
#(500, 500): 109 images
#(400, 500): 86 images
#(500, 335): 83 images


#--------------- check for Corrupted dat 
"""
corrupted_images = []

for i, example in enumerate(train):
    try:
        image = example["image"]
        image.verify() if image.format else image.load()
    except Exception:
        corrupted_images.append(i)

print("Total images checked:", len(train))
print("Corrupted images found:", len(corrupted_images))
print("Corrupted image indices:", corrupted_images[:20])
"""

#discover classes

class_counts = Counter(train["label"])

print("Number of classes:", len(class_counts))
print("Minimum images per class:", min(class_counts.values()))
print("Maximum images per class:", max(class_counts.values()))

labels = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(14, 5))
plt.bar(range(len(labels)), counts)
label_feature = train.features["label"]
if hasattr(label_feature, "names"):  # only use species names if the dataset provides them
    plt.xticks(range(len(labels)), [label_feature.names[l] for l in labels], rotation=90, fontsize=4)
    plt.xlabel("Bird species")
else:
    plt.xlabel("Bird class index")
plt.ylabel("Number of images")
plt.title("Training Dataset Class Distribution")
plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


#check for duplicates 
"""

image_hashes = []

for example in train:
    buffer = io.BytesIO()
    example["image"].save(buffer, format="PNG")
    image_hashes.append(
        hashlib.md5(buffer.getvalue()).hexdigest()
    )

hash_counts = Counter(image_hashes)
duplicate_groups = sum(
    count > 1 for count in hash_counts.values()
)
duplicate_extra = sum(
    count - 1 for count in hash_counts.values()
    if count > 1
)

print("Images checked:", len(train))
print("Duplicate groups:", duplicate_groups)
print("Extra duplicate records:", duplicate_extra)
"""



#[what_to_keep  for item in list  if condition] , data feha /n , /t 2st5dm strip
"""

record = train[0]

print("Bird species:", record["label"])
print("\nRaw description:")
print(record["description"])

captions = record["description"].splitlines()
captions = [caption.strip() for caption in captions if caption.strip()]

print("\nNumber of non-empty captions:", len(captions))

for i, caption in enumerate(captions, start=1):
    print(f"{i}. {caption}")
"""


#notes 1- remove 121->131 , 2- remove corrupted check (useless) , 3- dublication count (too much overhead for one 1 dublicate)

all_captions = []
empty_records = 0

for example in train:
    description = example["description"]

    captions = [
        caption.strip()
        for caption in description.splitlines()
        if caption.strip()
    ]

    if not captions:
        empty_records += 1

    all_captions.extend(captions)

word_counts = [len(caption.split()) for caption in all_captions]

words = Counter(
    word
    for caption in all_captions
    for word in re.findall(r"\b[a-z]+\b", caption.lower())#word tokenizer
)

print("Total images:", len(train))
print("Total non-empty captions:", len(all_captions))
print("Images with no non-empty captions:", empty_records)
print("Average words per caption:", round(sum(word_counts) / len(word_counts), 2))
print("Shortest caption:", min(word_counts), "words")
print("Longest caption:", max(word_counts), "words")

print("\n20 most common words:")
print(words.most_common(20))

plt.figure(figsize=(8, 5))
plt.hist(word_counts, bins=20)
plt.xlabel("Words per caption")
plt.ylabel("Number of captions")
plt.title("Caption Length Distribution")
plt.savefig("caption_length_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#m3loma leko ( extend vs append )
"""
append vs extend:

a = ["x"]
a.append(["y", "z"])
# ["x", ["y", "z"]]   <- adds the whole list as ONE item

b = ["x"]
b.extend(["y", "z"])
# ["x", "y", "z"]     <- adds each item separately
"""

#resize of images

#this resize is brute force (not good)
""" test for 1
image = train[0]["image"].convert("RGB")

resize_transform = transforms.Resize((224, 224))
resized_image = resize_transform(image)

print("Original size:", image.size)
print("Resized size:", resized_image.size)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(resized_image)
plt.title("Resized: 224 × 224")
plt.axis("off")

plt.show()
"""

resize_transform = transforms.Resize((224, 224))

X_train = np.empty((len(train), 224, 224, 3), dtype=np.uint8)

for i, example in enumerate(tqdm(train, desc="Resizing training images")):
    image = example["image"].convert("RGB")
    image = resize_transform(image)
    X_train[i] = np.asarray(image)

y_train = np.asarray(train["label"])

# same resize for test, no augmentation
X_test = np.empty((len(test), 224, 224, 3), dtype=np.uint8)

for i, example in enumerate(tqdm(test, desc="Resizing test images")):
    image = example["image"].convert("RGB")
    image = resize_transform(image)
    X_test[i] = np.asarray(image)

y_test = np.asarray(test["label"])

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)

# sanity checks: shapes, dtype, and image/label order
assert X_train.shape == (len(train), 224, 224, 3) and X_train.dtype == np.uint8
assert X_test.shape == (len(test), 224, 224, 3) and X_test.dtype == np.uint8
assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
for i in (0, len(test) // 2, len(test) - 1):
    assert y_test[i] == test[i]["label"]

# اقتراح مني بس لسا مجربتوش resize + padding 


#augmentation flip , mirrio , tilit by 10 degress and chagne brightness
# train only: applied on the fly to each batch during training (not saved), test stays un-augmented


augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15)
])

image = Image.fromarray(X_train[0])
augmented_image = augmentation(image)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Augmented")
plt.axis("off")

plt.show()

#note : tilt leaves a black pixels مش عارف صح ولا المفروض اعملها حاجه