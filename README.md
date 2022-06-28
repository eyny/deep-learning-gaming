# deep-learning-gaming
This is the graduation project I prepared in school to get my bachelor's degree in computer engineering. It learns how to play Super Mario Bros. game using deep reinforcement learning which can be considered a subset of machine learning. It uses [Keras](https://github.com/keras-team/keras) for its deep learning library with [TensorFlow](https://github.com/tensorflow/tensorflow) backend. [Gym Super Mario](https://github.com/ppaquette/gym-super-mario) environment bundle is used for [OpenAI Gym](https://github.com/openai/gym).
At the first stage, the best model from three different models for this problem is founded. During the second stage, the model is trained on the first level of Super Mario Bros for a long duration. At the last stage, the trained model is used on a different level. It is investigated if there is a significant improvement compared to a random model.

## Requirements
* Keras 2.1.5
* FCEUX 2.2.2 
* Gym Super Mario 0.0.7
* OpenAI Gym 0.92
