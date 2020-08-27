import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


# Function to define parser and its arguments
def define_arguments():
    
    # Create parser object
    parser = argparse.ArgumentParser(description = 'Flowers Image Classifier')

    # For each argument we provide(Name, Default Value, Data Type, and Help Message)

    # Image variable to get the image path used in the prediction
    parser.add_argument('--image', 
                    default = './test_images/wild_pansy.jpg', 
                    type = str, 
                    help = 'Image path')

    # Model variable to get the model path used for the prediction
    parser.add_argument('--model', 
                    default = './flowers_classifier.h5', 
                    type = str, 
                    help = 'Model file path')

    # K variable represents the top K most likely flowers
    parser.add_argument('--top_k', 
                    default = 5 , 
                    type = int, 
                    help = 'The top K most likely classes')

    # Variable to get the Json file that contains the labels
    parser.add_argument('--category_names', 
                    default = './label_map.json', 
                    help = 'Json file used to map the labels with category names')
    
    return parser
    
    
    
# Function that handles the processing of flower image before being injected to the prediction model
# ( Resizing and Normalizing )
def process_image(image):
    
    # Convert (NumPy array) image into a TensorFlow Tensor
    processed_image = tf.convert_to_tensor(image)
    # Resize the image
    processed_image = tf.image.resize(image, (224, 224))
    # Normalize the pixel values
    processed_image /= 255
    
    # Return the image as NumPy array
    return processed_image.numpy() 



# Function handles the prediction of labels by taking as an inputs (image path, loaded model, top K as an integer)
def predict(image_path, model, top_k):
    
    # First: Process the Image
        #1. Load and import the image
    image = Image.open(image_path)
        #2. Convert it to numpy array
    image = np.asarray(image)
        #3. Resize and normalize the image
    image = process_image(image)
        #4. Add Extra dimension represents the batch size, to make the image in the needed dimensions for the model
    image = np.expand_dims(image, axis = 0)
    
    # Second: Predict tha labels using the loaded model
    predicted_probabilities = model.predict(image)
    
    # Third: Interpret the results returned by the model    
        # Finds the k largest entries in the probabilities vector and outputs their values and crossoponding labels
    propabilities, classes = tf.nn.top_k(predicted_probabilities, k = top_k)
    
        # Converts both the probabilities and classes to numpy list of 1-D
    propabilities = propabilities.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]
    
    # Forth: Map the classes with the labels
    labels = []
    for l in classes:
        labels.append(class_names[str(l+1)]) # (+1) for the difference in the labels names
        
    return propabilities, labels
    
    
if __name__=="__main__":
    
    parser = define_arguments()

    arg_parser = parser.parse_args()

    # Save user inputs to variables
    image_path = arg_parser.image
    model_path = arg_parser.model
    top_k = arg_parser.top_k
    category_names = arg_parser.category_names


    # Load and map the labels to the flowers category
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    # Load the prediction model using TensorFlow 
    model = tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer})

    print("****Start Pridiction****\n")
    
    # Predict by passing the image path + loaded model + top k as integer
    probs, labels = predict(image_path, model, top_k)
    

    # Print the result of prediction 
    print("Top {} prediction flower names and it's associated probability for the image in path: {}\n".format(top_k, image_path))
    print('\t Flower Name | Probability% \n')
    
    for p, l in zip(probs, labels):
        p = float(format(p, '.4f'))
        print('\t {} | {}%'.format(l, p*100))
        
        
    print("\n****End Pridiction****")
