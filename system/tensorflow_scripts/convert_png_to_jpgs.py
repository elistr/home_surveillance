from PIL import Image
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
subdir = "people_files/people_photos/"
IMAGE_DIR = os.path.join(fileDir, subdir)


def convert_to_jpg(png_path, jpg_path):
    im = Image.open(png_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpg_path)


def get_png_paths(image_dir):
    res = []
    sub_dirs = [os.path.join(image_dir, o) for o in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, o))]
    for dir in sub_dirs:
        res.extend([os.path.join(dir, o) for o in os.listdir(dir)
                    if o.endswith(".png")])
    return res


def main():
    png_paths = get_png_paths(IMAGE_DIR)
    for png in png_paths:
        output_jpg_path = png[:-3]+"jpg"
        convert_to_jpg(png, output_jpg_path)


if __name__ == "__main__":
    main()
