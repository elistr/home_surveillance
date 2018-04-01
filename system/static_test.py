from tensorflow_scripts.label_image import run_label

fname = "/host/hacks/face_img.jpeg"

predictions = run_label(fname)

print(predictions[0])
