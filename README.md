# Handwritten-Text-Recognition-using-Computer-Vision-and-Deep-Learning

Digitized documents offer several advantages over physical documents, making accurate, efficient, and easy digitization of handwritten documents crucial. To automate the digitization of documents at ASU, we leveraged computer vision and deep learning. To achieve this, we followed several steps:

Data Acquisition: We used the IAM dataset for training and testing our model.<br>
Data Processing: This included normalizing pixel values, converting images to grayscale, and resizing and scaling all images to the same size.<br>
Data Augmentation: This step was essential to train the model to recognize characters in less-than-ideal situations, which is common in handwriting.<br>
Model Building: We built our model using a 1-D Convolutional Neural Network (CNN).<br>

Our model achieved a character error rate of 12.5%, demonstrating its effectiveness in recognizing handwritten text.

This end-to-end solution lifecycle effectively digitizes handwritten documents, providing an efficient and accessible way to handle important records.

