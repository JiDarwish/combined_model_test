import torch
import torch.nn as nn


# load the pre-trained image captioning model
captioning_model = torch.load("captioning_model.pt")
captioning_model.eval() # set the model in evaluation mode

# load the pre-trained image colorization model
colorization_model = torch.load("colorization_model.pt")
colorization_model.eval() # set the model in evaluation mode


# To test it's working
# image_captioning_output = image_captioning_output.reshape(3, 256, 256) # Maybe shoud use this to make sure it has the
# # same output shape
# #passing an example through the image captioning model
# image_captioning_output = captioning_model(example_input)
# print("Image Captioning Output Shape:", image_captioning_output.shape)
#
# #passing an example through the image colorization model
# image_colorization_output = colorization_model(image_captioning_output)
# print("Image Colorization Output Shape:", image_colorization_output.shape)


# create a new model that combine the captioning and colorization models
combined_model = nn.Sequential(captioning_model, colorization_model)

# set the optimizer
optimizer = torch.optim.Adam(combined_model.parameters(), lr=1e-5)
