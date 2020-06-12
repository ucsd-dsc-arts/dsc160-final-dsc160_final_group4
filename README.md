# Comparing Van Gogh’s paintings with Computer Generated Images in His Style

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Yicen Ma, yim095@ucsd.edu
- Xingyu Jiang, xij080@ucsd.edu
- Chang Yuan, chy238@ucsd.edu
- Michael Kusnadi, mkusnadi@ucsd.edu
- Kaixin Huang, k3huang@ucsd.edu

## Abstract

For the final project, we will find if there exists any differences or similarities between computer generated Van Gogh paintings and his actual work by using deep photo transfer and style transfer. This means that we will find some real images which has similar objects as some Van Gogh's paintings. Then we will put those real images into models and find the similarties or differences between computer generated painting and his actual work. 

We will divide 2 groups of real image in this project. One group we will put those real images into both deep photo transfer and style transfer. Another group we will just put real images into the style transfer. The reason for us to do that is we find out some real images that has pretty similar object and environment with Van Gogh's real painting. But their main color is different. So we use some strong color image as style and change their color by using deep photo transfer. Then putting them into style transfer and compare the difference and similarities between generated paintings and his actual work. Another group we will just put those real images into style transfer. And compare the differences and similarities between generated paintings and his actual work. 

For the result, I hope the deep photo transfer, it can successfully change the color from the style real image into the content real image. For the style transfer, it can accurately get Van Gogh's painting skills and changing our real images by using those painting skills. By making some images “van Gogh” like, we can better understand van Gogh’s artistic styles and how he would draw the objects that he had never tried to draw. And we hope our generated results from both groups can be some paintings that will let people think they are looking actual paintings from Van Gogh. 


Reference:

1. https://www.theverge.com/2020/4/2/21204498/art-transfer-google-artists-style-photos.
This is talking about using the art transfer tech by Google to let you apply famous artists’ styles to your own works

2. https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
This gives us some idea on Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution

3. http://headforart.com/2016/12/16/how-artists-use-colour/
This is talking about the importance of color for the artists. And this gives us the idea that we can change the color of those artists’ artworks and then put them into our Van-Gogh model for transferring to Van Gogh art style. And we want to see the influence on changing color on the original artworks during our process of transferring to Van Gogh style. (see whether it will generate different artworks after transfering) 

4. https://github.com/simulacre7/tensorflow-IPythonNotebook/blob/master/neural-style/neural_style.ipynb
This is a github project for combing one paint style into another artwork. We believe we will do the similar steps as this project.



## Data and Model
### Model:

Image Style Transfer Using Convolutional Neural Networks (CNN). This method inverts the image representation based on CNN and by using a texture model, it transfers the style of an image to another with an adjustable weight ratio of the two inputs that can affect the representation of the result.

Code: https://github.com/roberttwomey/dsc160-code/blob/master/examples/neural-style-transfer.ipynb

Image Style Transfer Using Convolutional Neural Networks: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf


Deep Photo Style Transfer, a method to apply the style of an image onto a content image but in realistic settings. With the use of segmentation masks, deep photo style transfer can apply style onto the specified objects instead of on random objects to make the result look realistic. 

Code: https://github.com/LouieYang/deep-photo-styletransfer-tf

Deep Photo Style Transfer - https://arxiv.org/pdf/1703.07511.pdf


### Training Data:

Artist Info

Van Gogh was a Dutch Post-Impressionist artist in the 1900 century. He was famed for his bold, dramatic brush strokes which expressed emotion and added a feeling of movement to his works.His emerging style saw him emotionally reacting to subjects through his use of color and brush work. He deliberately used colors to capture mood by using the complementary color contrasts and a bolder composition in his advanced years.


Images and oil painting:
	We selected all the images and oil paintings of landscapes.
	

Irises

The oil painting Irises was painted in 1889 inSaint Remy de Provence (France) and it showed the beauty of irises from a special point of view. Van Gogh used a high concentration of green and blue in this painting. It shows the full of softness and lightness. Thoses irises are  full of life without tragedy.
Similar Composition image: real image with irises flower.

Data Source:https://www.wikiart.org/en/vincent-van-gogh/irises-1889
https://longislandnatives.com/products/iris-versicolor-blue-flag-iris


Road with Cypress

The oil painting Road with Cypress was painted in 1890 in Saint Remy de Provence (France) and it showed a tall cypress tree in a country side. The sky in this painting shares the similar sky in one of the famous paintings The Starry Night. Also, the cypress was always presented in Van Gogh’s paintings in the advanced years. The Cypress dominated the painting and dwarfed elements around it.
Similar Composition image: it also has a road with a cypress. However, the land was covered by snow, because we want to use the deep photo transfer to add the color on it.
Deep photo transfer image: it is a landscape of a wheatfield in yellow and cloudy in the sky.

Data Source:https://www.wikiart.org/en/vincent-van-gogh/road-with-cypresses-1890	https://www.123rf.com/photo_36085575_lonely-cypress-tree-and-snow-in-winter-season-rural-landscape-val-d-orcia-tuscany-italy.html
https://commons.wikimedia.org/wiki/File:Wheatfield_in_Ottawa.jpg


Summer Evening, Wheatfield with Setting sun

Van Gogh painted the oil painting Summer Evening, Wheatfield with Setting sun in 1888 in France. There were vast tracts of wheatfield in the painting with the village and sunset as the background. He used different linewidth to represent each wheat from near to far.
Similar Composition image: it is a landscape of a wheatfield in yellow and cloudy in the sky. There are also three buildings in shadow in the distance.

Data Source: https://www.wikiart.org/en/vincent-van-gogh/summer-evening-wheatfield-with-setting-sun-1888
https://commons.wikimedia.org/wiki/File:Wheatfield_in_Ottawa.jpg


Sunflowers

The oil painting Still Life - Vase with Fifteen Sunflowers was painted in 1888 in France. Van Gogh painted a sunflower series and there were only some minor differences between each oil painting. We selected the all yellow color in the background, vase and sunflower itself with green in stem. But he used different brightness of yellow to distinguish the edges.
Similar Composition image: a landscape of polar forest fulfilled with fallen leaves in orange color.

Data Source: https://www.wikiart.org/en/vincent-van-gogh/still-life-vase-with-fifteen-sunflowers-1888-1
http://image.baidu.com/search/detail?ct=503316480&z=undefined&tn=baiduimagedetail&ipn=d&word=秋天落叶&step_word=&ie=utf-8&in=&cl=2&lm=-1&st=undefined&hd=undefined&latest=undefined&copyright=undefined&cs=3998557001,4033328731&os=3489487148,3249175063&simid=3472044273,490966350&pn=12&rn=1&di=50490&ln=1694&fr=&fmq=1591717961456_R&fm=&ic=undefined&s=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&is=0,0&istype=0&ist=&jit=&bdtype=0&spn=0&pi=0&gsm=0&objurl=http%3A%2F%2Fimage.biaobaiju.com%2Fuploads%2F20181223%2F18%2F1545559990-gOTFWpYrZf.jpg&rpstart=0&rpnum=0&adpicid=0&force=undefined&ctd=1591717967400^3_1440X837%1
https://www.52112.com/pic/143913.html


The Church at Auver

When Van Gogh was at Auver, he saw a church and painted in 1890 at Auver in France. It reminded him back to the landscape of  his childhood with small houses with thatched roofs. This time, he didn't paint the sky as dart blue color rather than the style in The Starry Night. It brought a peaceful emotion from this painting.
Similar Composition image: This is the church prototype from Van Gogh’s painting at Auver. In 1890, Van Gogh saw the church and created The Church at Auver.

Data Source: https://www.wikiart.org/en/vincent-van-gogh/the-church-at-auvers-1890
	https://www.wikiwand.com/en/The_Church_at_Auvers

## Code
### Deep Photo Style Transfer part of code: 

Description of deep photo style transfer: The execution of the deep photo style transfer method is a little different than usual, in that to run this code, you have to run the specified command line in the console. The .py files necessary to run the command line are listed above, and the exact command lines are given in the Deep_Photos_Style_Transfer.ipynb.

1. https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/Deep_Photos_Style_Transfer.ipynb

	This notebook contains the necessary commands needed to run the deep photo style transfer method onto the specified content image and the style image. 

2. https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/closed_form_matting.py

	This py file uses the Matting Laplacian to constrain the transformation from the input to the output to be locally affine in colorspace.
	
3. https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/deep_photostyle.py

	This py file contains all the argument input options such as the training optimizer, the weight regularization, and whether to apply the smooth local affine. Once all the necessary options are specified, this deep_photostyle py file will execute photo_style.py, closed_form_matting.py, and smooth_local_affine.py if specified. 

4. https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/photo_style.py

	This py file calls the loss function that tries to minimize the content loss and style loss. This enables the style transfer affect, which is overlaying the style of the style image onto the content image. In addition, the photorealism effect can be augmented by incorporating segmentation masks by calling the stylized function. This helps to specify which style certain objects should contain instead of randomly over-laced with random colors. 

5. https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/smooth_local_affine.py

	This py file enables the reconstructed image to be represented by locally affine color transformations of the input to prevent distortions. 


### Neural Style Transfer part of code: 
https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/Code/neural_style_transfer.ipynb

Description of neural style transfer: 
This method transfers the style of images using Convolutional Neural Networks (CNN). The notebook from the link above contains two parts: data scraping and preprocessing, and execution of style transfer. For the first part of the notebook, we scraped the images we need from wikiart. Then, along with the content images we had locally, we scaled and cropped them so that the content images and style images are in the same size for style transfer (we did this for each set of content image and style image, and for two sets of them we did the style transfer on the images we got from deep photo transfer). We used gpu to make the images have higher resolutions. After transforming them to pytorch tensors, we are done with the first part. For the second part, we used the pretrained vgg19 model to make the style transfer while tracking the style losses and content losses. And by trying different combinations of style weights and content weights, we choosed the best resulting images as our final results.

## Results

result link:
https://github.com/ucsd-dsc-arts/dsc160-final-dsc160_final_group4/blob/master/results/PDF%20version%20of%20result%20part%20(1).pdf


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

Chang Yuan: any code related work of neural style transfer and its results

Xingyu Jiang: idea thinking, data finding, proposal and first draft result part 

Yicen Ma: idea generator, idea thing, data finding, data part, proposal and final writing of result part

Michael Kusnadi: discussion part

Kaixin Huang: any code related work of deep photo style transfer and its results


## Technical Notes and Dependencies

Neural Style Transfer:
We mainly used pytorch for this part of the code. And we used the pretrained vgg19 model as the Convolutional Neural Networks (CNN) for our work. Other than that, we used some other standard libraries such as numpy, matplotlib, beautifulsoup, etc. The code can be run on datahub and no other software is required.

For rapid iterative development which is crucial for deep photo style transfer, PYCUDA is recommended for installation. This enables NVIDIA GPU accelerated computing with Python.


## Reference

https://github.com/roberttwomey/dsc160-code/blob/master/examples/scrape-wikiart.ipynb

https://github.com/roberttwomey/dsc160-code/blob/master/examples/neural-style-transfer.ipynb

https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

https://www.theverge.com/2020/4/2/21204498/art-transfer-google-artists-style-photos.

https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398

http://headforart.com/2016/12/16/how-artists-use-colour/

https://github.com/simulacre7/tensorflow-IPythonNotebook/blob/master/neural-style/neural_style.ipynb

https://github.com/LouieYang/deep-photo-styletransfer-tf

https://arxiv.org/pdf/1703.07511.pdf

