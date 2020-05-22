# Project Title

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Yicen Ma, yim095@ucsd.edu
- Xingyu Jiang, xij080@ucsd.edu
- Kaixin Huang, k3huang@ucsd.edu
- Firstname Lastname4, name4@ucsd.edu
- Firstname Lastname5, name5@ucsd.edu

## Abstract

 For the final project, we want to transfer a couple of the most famous paintings by using the tensorflow algorithm. Then, we add Van Gogh oil painting style on to those images to see how those famous paintings changed. For the methods and techniques, we want to use the deep photo style tensorflow, neural style, generative adversarial networks to generate the final result. Firstly, we will scrape all the pictures from Van Gogh and select the oil paintings from Van Gogh. Secondly, we will choose some representative artworks from other famous artists, for example Leonardo da Vinci, and select a couple images. Then, we divide 2 groups, one is transfer those images to process the deep photo style and the other one is not doing any transformation. Thirdly, we will combine the paintings with the style from Van Gogh’s to generate 2 new groups of paintings and contrast the new paintings from 2 groups. For the results, we hope the system will pick the traits of Van Gogh’s artworks and then transfer those representative artworks from other famous artists and other real-world photos to Van Gogh style artworks. Our results will be images of those famous paintings in Van Gogh style.

By using all the artworks, it would be challenging to truly define his art style due to his many works. It is hard for us to pick a suitable painting and we want to make sure the result is not awkward. To expand the topics from the lectures, we will continue the research for lecture 13 of the style transfer and deep photo style transfer. In our project, we will first process the deep photo style transfer and then process the style transfer. For the training data we select, we will have 2 groups, one is the original version and the other version is using the deep photo style transfer. Then we put both versions into the style transfer and both transfer to Van-gogh style. And then we will see whether there exist any differences between 2 groups. We will compare those results on the differences and focus on how the traits we add are special in the painting. The interesting part about this project is that we can see some of the most famous paintings in the world in van Gogh’s style, which can be interesting as there can be some interesting combinations of styles and objects. By making some images “van Gogh” like, we can better understand van Gogh’s artistic styles and how he would draw the objects that he had never tried to draw.
Reference:

1.https://www.theverge.com/2020/4/2/21204498/art-transfer-google-artists-style-photos.
This is talking about using the art transfer tech by Google to let you apply famous artists’ styles to your own works

2.https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
This gives us some idea on Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution

3.http://headforart.com/2016/12/16/how-artists-use-colour/
This is talking about the importance of color for the artists. And this gives us the idea that we can change the color of those artists’ artworks and then put them into our Van-Gogh model for transferring to Van Gogh art style. And we want to see the influence on changing color on the original artworks during our process of transferring to Van Gogh style. (see whether it will generate different artworks after transfering) 

4.https://github.com/simulacre7/tensorflow-IPythonNotebook/blob/master/neural-style/neural_style.ipynb
This is a github project for combing one paint style into another artwork. We believe we will do the similar steps as this project.



## Data and Model

(10 points) 

In the final submission, this section will describe both the data you use for this project and any pre-existing models/neural nets. For each you should provide the name, a textual description, and a link. If there is a paper (for neural net) link that as well.
- Such and such Neural Net. The short description of this neural net. 
  - [link to code]().
  - [Title of Paper with Link](). 
- Training data. Short description of training data including bibliographic info. [link to data]().

## Code

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for: 

- code for data acquisition/scraping
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

## Results

(30 points) 

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below: 

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

## Discussion

(30 points, three to five paragraphs)

The first paragraph should be a short summary describing your results.

The subsequent paragraphs could address questions including:
- Why is this culturally innovative?
- How does your generative computational approach differ from traditional art/music/cultural production? 
- How do your results relate to broader social, cultural, economic political, etc., issues? 
- What are the ethical concerns for this form of generative art? 
- In what future directions could you expand this work?

## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.

## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
- Blog posts
