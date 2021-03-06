IMAGE_SIZE=128
ARCHITECTURE="Inceptionv3"
# ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"



python -m scripts.retrain \
  --bottleneck_dir=people_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=people_files/models/ \
  --summaries_dir=people_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=people_files/retrained_graph.pb \
  --output_labels=people_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=people_files/people_photos/
