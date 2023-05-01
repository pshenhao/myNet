# myNet
My project of graduation paper.

# How to use the project?
In this project, We need download the dataset in the link: "https://pan.baidu.com/s/1apd-CDDfrtMe22M9auW9Zw ,which password is : pshh, and save it in the same path of code files. We create three folder named "data", "record", and "saved_images" to insure the code will run. The folder named data is used to provide dataset, and others will save the training reasult of the model. We can also alter the paramer of the path in the training file.

# How to train?
The main training code is in the file named "train.py" and "train_5fold.py", and the main difference of both is whether use 5-fold in the training cycle. We can choose the model in code by alter the paramer named MODEL_KEY. The result of all model will save in the file, and wo needn't alter the file path.

# The main work?
I have added Inception, Outline Kernel and Attention in the U-Net, and the code was written in file named model0_myUNet. In other files, I use the name to show the function of the file.

