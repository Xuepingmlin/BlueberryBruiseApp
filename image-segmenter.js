/* global tf, Image, FileReader, ImageData, fetch */

const ImageCropUrl = 'model/ImageCrop/model.json'
const berrySegUrl = 'model/berrySegmentation/model.json'
const bruiseSegUrl = 'model/bruiseSegmentation/model.json'

// const ImageCropUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/ImageCrop/model.json'
// const berrySegUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/berrySegmentation/model.json'
// const bruiseSegUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/bruiseSegmentation/model.json'


const colorMapUrl = 'assets/color-map.json'

//const imageSize = 512
const imageSize = 224

let targetSize = { w: imageSize, h: imageSize }
let model
let imageElement
let colorMap
let output


/**
 * load the TensorFlow.js model
 */
window.loadModel = async function () {
  // disableElements()
  message('loading model...')

  let start1 = (new Date()).getTime()
  // https://js.tensorflow.org/api/1.1.2/#loadGraphModel
  ImageCrop_model = await tf.loadGraphModel(ImageCropUrl)
  let end1 = (new Date()).getTime()
  //message(ImageCrop_model.ImageCropUrl)
  message(`ImageCrop_model loaded in ${(end1 - start1) / 1000} secs`, true)

  let start2 = (new Date()).getTime()
  berrySeg_model = await tf.loadGraphModel(berrySegUrl)
  let end2 = (new Date()).getTime()
  //message(berrySeg_model.berrySegUrl)
  message(`berrySeg_model loaded in ${(end2 - start2) / 1000} secs`, true)

  let start3 = (new Date()).getTime()
  bruiseSeg_model = await tf.loadGraphModel(bruiseSegUrl)
  let end3 = (new Date()).getTime()
  //message(bruiseSeg_model.bruiseSegUrl)
  message(`bruiseSeg_model loaded in ${(end3 - start3) / 1000} secs`, true)
  document.getElementById('modelimage').disabled = false
  // enableElements()
}

// function disable_botton(){
//   document.getElementBy('modelrun').disabled = true
//   document.getElementBy('imagecrop').disabled = true
//   document.getElementBy('berrysegment').disabled = true
//   document.getElementBy('bruisesegment').disabled = true
// }


originalImage=new Image()
/**
 * handle image upload
 *
 * @param {DOM Node} input - the image file upload element
 */
window.loadImage = function (input) {
  let canvas_1= document.getElementById("canvasimage")
  canvas_1.getContext('2d').clearRect(0,0,canvas_1.width,canvas_1.height);
  let canvas_2= document.getElementById("canvascrop")
  canvas_2.getContext('2d').clearRect(0,0,canvas_2.width,canvas_2.height);
  let canvas_3 =document.getElementById("bruiseResult")
  canvas_3.getContext('2d').clearRect(0,0,canvas_3.width,canvas_3.height);
  let canvas_4 =document.getElementById("origin")
  canvas_4.getContext('2d').clearRect(0,0,canvas_4.width,canvas_4.height);
  let canvas_5 =document.getElementById("berrySegmentation")
  canvas_5.getContext('2d').clearRect(0,0,canvas_5.width,canvas_5.height);
  let canvas_6 =document.getElementById("bruiseSegmentation")
  canvas_6.getContext('2d').clearRect(0,0,canvas_6.width,canvas_6.height);

  if (input.files && input.files[0]) {
    // console.log('input:'+input.files[0])
    // disableElements('input:'+input.files )
    message('resizing image...')

    let reader = new FileReader()

    reader.onload = function (e) {
      let src = e.target.result

      document.getElementById('canvasimage').getContext('2d').clearRect(0, 0, targetSize.w, targetSize.h)
      // document.getElementById('canvassegments').getContext('2d').clearRect(0, 0, targetSize.w, targetSize.h)

      imageElement = new Image()
      imageElement.src = src
      originalImage=imageElement.cloneNode();

      imageElement.onload = function () {
        //let resizeRatio = imageSize / Math.max(imageElement.width, imageElement.height)
        //targetSize.w = Math.round(resizeRatio * imageElement.width)
        //targetSize.h = Math.round(resizeRatio * imageElement.height)
        let resizeRatio1 = imageSize / imageElement.width
        let resizeRatio2 = imageSize / imageElement.height
        targetSize.w = Math.round(resizeRatio1 * imageElement.width)
        targetSize.h= Math.round(resizeRatio2 * imageElement.height)

        let origSize = {
          w: imageElement.width,
          h: imageElement.height
        }
        imageElement.width = targetSize.w
        imageElement.height = targetSize.h

        let canvas = document.getElementById('canvasimage')
        // canvas.width = targetSize.w/2
        // canvas.height = targetSize.h/2
        let ctx=canvas.getContext('2d')
        // ctx.font='15px Arial'
        // ctx.fillText('resized original image:',0,15);
        ctx.drawImage(imageElement, 0, 0,targetSize.w/2,targetSize.h/2)

        // canvas
        //   .getContext('2d')
        //   .fillText('resized original image',0,0)
        //   .drawImage(imageElement, 19, 38,targetSize.w/2,targetSize.h/2)

        message(`resized from ${origSize.w} x ${origSize.h} to ${targetSize.w} x ${targetSize.h}`)
      }
    }

    reader.readAsDataURL(input.files[0])
  } else {
    message('no image uploaded', true)
  }
  document.getElementById('modelrun').disabled = false
}

/**
 * run the model and get a prediction
 */
individualBerry=[];
var berry_resize_ratio
window.runModel = async function () {
  message('running cropping...')
  let start = (new Date()).getTime()
  if (imageElement) {
    let img = preprocessInput(imageElement)
    // https://js.tensorflow.org/api/latest/#tf.Model.predict
    const output = ImageCrop_model.predict(img)
    // let end = (new Date()).getTime()
    segmentData= await processOutput(output,[128,0,0],175)
    // message(`inference ran in ${(end - start) / 1000} secs`, true)
    // message(`finish cropping`)
    document.getElementById('berrysegment').disabled = false
  } else {
    message('no image available', true)
  }

  individualBerry=[];
  let Ratio1 = imageSize / originalImage.width/2
  let Ratio2 = imageSize / originalImage.height /2
  let src_mask = cv.matFromImageData(segmentData);
  let dst_mask= new cv.Mat.zeros(src_mask.rows,src_mask.cols, cv.CV_8UC3);
  cv.cvtColor(src_mask, src_mask, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(src_mask, src_mask, 20, 100, cv.THRESH_BINARY);
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(src_mask, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
  Rect=[];
  for (let i = 0; i < contours.size(); i++) {
    let cnt = contours.get(i);
    let rect = cv.boundingRect(cnt);
    // console.log('i:'+i+'rect:'+rect.x+'*'+ rect.y+'*'+rect.width+'*'+rect.height)
    let contoursColor = new cv.Scalar(255, 255, 255);
    let rectangleColor = new cv.Scalar(255, 0, 0);
    cv.drawContours(dst_mask, contours, i, contoursColor, 1, cv.LINE_8, hierarchy, 100);
    let point1 = new cv.Point(rect.x, rect.y);
    let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
    cv.rectangle(dst_mask, point1, point2, rectangleColor, 0.1, cv.LINE_AA, 0)

    let rect_resize =Object.assign({},rect);
    rect_resize.x=(rect.x-1)/Ratio1;
    rect_resize.y=(rect.y-1)/Ratio2;
    rect_resize.width=(rect.width+1)/Ratio1;
    rect_resize.height=(rect.height+1)/Ratio2;
    Rect.push(rect_resize);
    }

  // order blueberries
  Rect.sort((a, b) => (a.y > b.y) ? 1 : -1 )
  var meanHeight = Rect.reduce(function(prev, cur) {return prev + cur.height; }, 0)/Rect.length;
  Rect_1=[];
  k_temp=[];
  RRect=[]
  k_temp.push(0)

  for (k=1;k<Rect.length;k++){
    if((Rect[k].y-Rect[k-1].y)>meanHeight/2){
      // console.log('k:'+k)
      k_temp.push(k);
    }    
  }
  // console.log('k_temp:'+k_temp)

  for (m=0;m<k_temp.length;m++){
    Rect_1=Rect.slice(k_temp[m],k_temp[m+1])
    Rect_1.sort((a, b) => (a.x > b.x) ? 1 : -1 )
    // console.log('Rect_1:'+Rect_1)
    for (n=0;n<Rect_1.length;n++){
      // console.log('n:'+n)
      RRect.push(Rect_1[n])
    }
  }
  // console.log('RRect:'+RRect)


  cv.imshow('canvascrop', dst_mask);
  src_mask.delete(); dst_mask.delete(); contours.delete(); hierarchy.delete();

  // extract orignial size berry
  var canvas_orgImg=document.createElement('canvas');
  var ctx_orgImg = canvas_orgImg.getContext('2d');
  canvas_orgImg.width = originalImage.width;
  canvas_orgImg.height = originalImage.height;
  ctx_orgImg.drawImage(originalImage, 0, 0 );
  let originalImageData = ctx_orgImg.getImageData(0, 0, canvas_orgImg.width, canvas_orgImg.height);
  let src_orgImg = cv.matFromImageData(originalImageData);
  let dst_orgImg= new cv.Mat();

  for (let j=0;j<RRect.length;j++){
    dst_orgImg=src_orgImg.roi(RRect[j]);
    individualBerry.push(dst_orgImg);
    // cv.imshow('individualBerry', dst_orgImg);
  }
  let end = (new Date()).getTime()
  message(`finish cropping in ${(end - start) / 1000} secs`, true)

  // document.getElementById('berrysegment').disabled = false
}


//berry segmentaion for individual berry
var bruise_ratio=[]
window.segmentBerry = async function () {
  message(`start segmentation...`)
  let start = (new Date()).getTime()

  let bruiseResult_canvas=document.getElementById('bruiseResult');
  let bruiseResult_ctx=bruiseResult_canvas.getContext('2d');
  bruiseResult_ctx.font = "20px Times New Roman";
  bruiseResult_ctx.fillText('' ,0, 0);
  // display bruise result
  for (var i=0;i<individualBerry.length;i++){
    let img=CreatImageData(individualBerry[i],224,224);
    let inputImage = preprocessInput(img);
    const output_berry = berrySeg_model.predict(inputImage);
    count_berry=CountPixel(output_berry);
    const output_bruise = bruiseSeg_model.predict(inputImage)
    count_bruise=CountPixel(output_bruise);
    if(count_bruise==0){
      ratio=0;
    } else{
      ratio=count_bruise/count_berry;
    };
    bruise_ratio.push(ratio)
    // console.log('i='+i+'  ratio:'+ratio)

    bruiseResult_ctx.fillText('id='+(i+1)+'  bruiseRatio='+ratio.toFixed(2) , 0, i*25+25);

    var width=individualBerry[i].cols;
    var height= individualBerry[i].rows;
    temp_ratio=4000/originalImage.height*0.15
    let Ori_berry=CreatImageData(individualBerry[i],width*temp_ratio,height*temp_ratio);
    let berry_imageData= await processOutput(output_berry,[255,255,255],150)
    let brusie_imageData= await processOutput(output_bruise,[255,0,0],150)
    let mat_berry = cv.matFromImageData(berry_imageData);
    let mat_bruise = cv.matFromImageData(brusie_imageData);
    let Ori_berry_mask=CreatImageData(mat_berry,width*temp_ratio,height*temp_ratio);
    let Ori_bruise_mask=CreatImageData(mat_bruise,width*temp_ratio,height*temp_ratio);
    if(i<10){
      drawImage('origin',Ori_berry,(i)*120,0)
      drawImage('berrySegmentation',Ori_berry_mask,(i)*120,0)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i)*120,0)
    } else if(i<20){
      drawImage('origin',Ori_berry,(i-10)*120,120)
      drawImage('berrySegmentation',Ori_berry_mask,(i-10)*120,120)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-10)*120,120)
    } else if (i<30){
      drawImage('origin',Ori_berry,(i-20)*120,240)
      drawImage('berrySegmentation',Ori_berry_mask,(i-20)*120,240)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-20)*120,240)
    } else if (i<40){
      drawImage('origin',Ori_berry,(i-30)*120,360)
      drawImage('berrySegmentation',Ori_berry_mask,(i-30)*120,360)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-30)*120,360)
    } else if (i<50) {
      drawImage('origin',Ori_berry,(i-40)*120,480)
      drawImage('berrySegmentation',Ori_berry_mask,(i-40)*120,480)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-40)*120,480)
    } else if (i<60) {
      drawImage('origin',Ori_berry,(i-50)*120,600)
      drawImage('berrySegmentation',Ori_berry_mask,(i-50)*120,600)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-50)*120,600)
    } else {
      drawImage('origin',Ori_berry,(i-60)*120,720)
      drawImage('berrySegmentation',Ori_berry_mask,(i-60)*120,720)
      drawImage('bruiseSegmentation',Ori_bruise_mask,(i-60)*120,720)
    };
  }
  let end = (new Date()).getTime()
  message(`finish segmentation in ${(end - start) / 1000} secs`, true)
}


function CreatImageData(ImageMat,rx,ry){
  var canvas=document.createElement('canvas');
  let dst = new cv.Mat();
  let dsize = new cv.Size(rx, ry);
  cv.resize(ImageMat, dst, dsize, 0, 0, cv.INTER_AREA);
  cv.imshow(canvas,dst);
  img=canvas.getContext('2d').getImageData(0,0,rx,ry)
  return img;
}

function drawImage(canvasid,imageData,x,y){
  var canvas=document.getElementById(canvasid)
  var ctx=canvas.getContext("2d");
  ctx.putImageData(imageData,x,y)
}


// function CreatImage(imgaeData){
//   var canvas=document.createElement('canvas');
//   canvas.getContext("2d").putImageData(imgaeData,0,0)
//   var img=document.createElement('img');
//   img.src = canvas.toDataURL("image/png");
//   document.body.appendChild(img);
// }





/**
 * convert image to Tensor input required by the model
 *
 * @param {HTMLImageElement} imageInput - the image element
 */
function preprocessInput (imageInput) {
  // console.log('preprocessInput started')

  let inputTensor = tf.browser.fromPixels(imageInput).toFloat()
  // console.log('inputTensor:'+inputTensor)

  // https://js.tensorflow.org/api/latest/#expandDims
  let preprocessed = inputTensor.expandDims()

  // console.log('preprocessInput completed:', preprocessed)
  return preprocessed
}

function CountPixel (output) {
  // console.log('processOutput started')

  let segMap = Array.from(output.dataSync())
  // console.log('segMap:'+segMap)
  var count=0;
  for (j=1;j<segMap.length;j=j+2){
    if(segMap[j]>0.8){
      count=count+1;
    }
  }
  return count;

}

/**
 * convert model Tensor output to image data for previewing
 *
 * @param {Tensor} output - the model output
 */
async function processOutput (output,color,a) {
  // console.log('processOutput started')

  let segMap = Array.from(output.dataSync())
  // console.log('segMap:'+segMap)
  segMapColor=[]
  for(var i=0;i<segMap.length;i++){
    if (segMap[i]>0.8){
      segMapColor.push(color)
    } else {
      segMapColor.push([0,0,0])
    }
  }


  // if (!colorMap) {
  //   await loadColorMap()
  // }
  // console.log('colorMap:'+colorMap)
  // let segMapColor = segMap.map(seg => colorMap[(seg>0.8)?1:0])
  // console.log('segMapColor:'+segMapColor)

  // let canvas = document.getElementById('canvassegments')
  let canvas = document.createElement('canvas')
  let ctx = canvas.getContext('2d')
  canvas.width = targetSize.w/2
  canvas.height = targetSize.h/2
  // console.log('segMaColor:'+segMapColor)
  let data = []
  for (var i = 1; i < segMapColor.length; i=i+2) {
    data.push(segMapColor[i][0]) // red
    data.push(segMapColor[i][1]) // green
    data.push(segMapColor[i][2]) // blue
    data.push(a) // alpha
    // data.push(color[0]) // red
    // data.push(color[1]) // green
    // data.push(color[2]) // blue
    // data.push(color[3]) // alpha
  }

  let imageData = new ImageData(canvas.width, canvas.height)
  imageData.data.set(data)
  ctx.putImageData(imageData, 0, 0)

  return imageData

  //console.log('processOutput completed:', imageData)

}


async function loadColorMap () {
  let response = await fetch(colorMapUrl)
  colorMap = await response.json()

  if (colorMap && colorMap.hasOwnProperty('colorMap')) {
    colorMap = colorMap['colorMap']
  } else {
    console.warn('failed to fetch colormap')
    colorMap = []
  }
}

function disableElements () {
  const buttons = document.getElementsByTagName('button')
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].setAttribute('disabled', true)
  }

  const inputs = document.getElementsByTagName('input')
  for (var j = 0; j < inputs.length; j++) {
    inputs[j].setAttribute('disabled', true)
  }
}

function enableElements () {
  const buttons = document.getElementsByTagName('button')
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].removeAttribute('disabled')
  }

  const inputs = document.getElementsByTagName('input')
  for (var j = 0; j < inputs.length; j++) {
    inputs[j].removeAttribute('disabled')
  }
}

function message (msg, highlight) {
  let mark = null
  if (highlight) {
    mark = document.createElement('mark')
    mark.innerText = msg
  }

  const node = document.createElement('div')
  // node.addEventListener('load', function() {
  //   console.log('message loaded');
  // });
  if (mark) {
    node.appendChild(mark)
  } else {
    node.innerText = msg
  }
  
  document.getElementById('message').appendChild(node)
  // document.getElementById('message')

}

function init () {
  message(`tfjs version: ${tf.version.tfjs}`, true)
}

// ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  setTimeout(init, 500)
}

