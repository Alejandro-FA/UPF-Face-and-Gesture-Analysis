import os
import numpy as np
import argparse
import cv2
import os

# NOTE: This code is not original, it was taken from https://pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/
# The original code was written by Adrian Rosebrock

def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicates in a dataset")
    parser.add_argument("--dataset", '-d', type=str, required=True, help="Path to the dataset")
    parser.add_argument("--remove", '-r', required=False, help="Whether or not duplicates should be removed (i.e., dry run)", action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
	# construct the argument parser and parse the arguments
    args = parse_args()
    dataset = args.dataset
    remove = args.remove

    # Get image paths
    image_filenames = [f for f in os.listdir(dataset) if f.endswith(".jpg")]
    image_paths = [os.path.join(dataset, f) for f in image_filenames]

    # grab the paths to all images in our input dataset directory and
    # then initialize our hashes dictionary
    print("[INFO] computing image hashes...")
    hashes = {}

    # loop over our image paths
    for imagePath in image_paths:
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image, hashSize=4)
        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

    # loop over the image hashes
    for (h, hashedPaths) in hashes.items():
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:

            # check to see if this is a dry run
            if remove is False:
                # initialize a montage to store all images with the same
                # hash
                montage = None
                # loop over all image paths with the same hash
                for p in hashedPaths:
                    # load the input image and resize it to a fixed width
                    # and heightG
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))
                    # if our montage is None, initialize it
                    if montage is None:
                        montage = image
                    # otherwise, horizontally stack the images
                    else:
                        montage = np.hstack([montage, image])
                # show the montage for the hash
                print("[INFO] hash: {}".format(h))
                for p in hashedPaths:
                    print(f"\t{p}")
                cv2.imshow("Montage", montage)
                cv2.waitKey(0)

            # otherwise, we'll be removing the duplicate images
            else:
                # loop over all image paths with the same hash *except*
                # for the first image in the list (since we want to keep
                # one, and only one, of the duplicate images)
                for p in hashedPaths[1:]:
                    os.remove(p)