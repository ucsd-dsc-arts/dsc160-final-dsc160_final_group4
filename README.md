# Project Title

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Yicen Ma, yim095@ucsd.edu
- Xingyu Jiang, xij080@ucsd.edu
- Firstname Lastname3, name3@ucsd.edu
- Firstname Lastname4, name4@ucsd.edu
- Firstname Lastname5, name5@ucsd.edu

## Abstract
For the final project, we will find if there exists any differences or similarities between computer generated Van Gogh paintings and his actual work by using deep photo transfer and style transfer. This means that we will find some real images which has similar objects as some Van Gogh's paintings. Then we will put those real images into models and find the similarties or differences between computer generated painting and his actual work. 

We will divide 2 groups of real image in this project. One group we will put those real images into both deep photo transfer and style transfer. Another group we will just put real images into the style transfer. The reason for us to do that is we find out some real images that has pretty similar object and environment with Van Gogh's real painting. But their main color is different. So we use some strong color image as style and change their color by using deep photo transfer. Then putting them into style transfer and compare the difference and similarities between generated paintings and his actual work. Another group we will just put those real images into style transfer. And compare the differences and similarities between generated paintings and his actual work. 

For the result, I hope the deep photo transfer, it can successfully change the color from the style real image into the content real image. For the style transfer, it can accurately get Van Gogh's painting skills and changing our real images by using those painting skills. By making some images “van Gogh” like, we can better understand van Gogh’s artistic styles and how he would draw the objects that he had never tried to draw. And we hope our generated results from both groups can be some paintings that will let people think they are looking actual paintings from Van Gogh. 
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
