# CV_Object_Recognition

This work is based on https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch. 

The original training and testing datasets in jpg format are from https://www.kaggle.com/datasets/tongpython/cat-and-dog?select=test_set

In this study, I delve into the exploration of using linear and logarithmic RGB data transformed from JPEG sRGB images as inputs to deep neural networks for computer vision tasks. Through iterative experimentation, architectural refinements, and considerations in data preprocessing, I aimed to achieve the best results in the CatsOrDogs task across all three data formats. The rationale behind selecting the CatsOrDogs task and dataset stems from the fact that the images within the dataset are captured under various lighting conditions. This selection offers a unique opportunity to gain valuable insights into the practical implications of employing different data representations, which either preserve or discard physics-based features present in the images, such as lighting and shadows.

The outcomes align with the hypothesis that using linear and log data transformed from sRGB data provides advantages to neural network performance, particularly in terms of testing and training accuracy, even when the original data is already in a compressed form. However, the anticipated result that log data, which retains the most physics features, would yield the highest performance was not achieved (due to model overfitting)

The result data spreadsheet is: https://docs.google.com/spreadsheets/d/1VtR-XxqIvcK20O3INQhlSxQtq4BIXc4N1uijRB0Kwfk/edit#gid=874896169

the report/paper I wrote for my summer research is: https://docs.google.com/document/d/1TgXQl4Jqb9YJxK2HTrP066I0pqyY7SKOEBA8KTenJNo/edit

Some future work ideas could be modifying learning rate, scheduler, and augmentation based on: https://datahacker.rs/016-pytorch-three-hacks-for-improving-the-performance-of-deep-neural-networks-transfer-learning-data-augmentation-and-scheduling-the-learning-rate-in-pytorch/
