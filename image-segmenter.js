/* global tf, Image, FileReader, ImageData, fetch */

const ImageCropUrl = 'model/ImageCrop/model.json'
const berrySegUrl = 'model/berrySegmentation/model.json'
const bruiseSegUrl = 'model/bruiseSegmentation/model.json'

// const ImageCropUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/ImageCrop/model.json'
// const berrySegUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/berrySegmentation/model.json'
// const bruiseSegUrl ='https://github.com/Xuepingmlin/BlueberryBruiseApp/blob/master/model/bruiseSegmentation/model.json'


const colorMapUrl = '/assets/color-map.json'

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
  if (input.files && input.files[0]) {
    console.log('input:'+input.files[0])
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
        canvas.width = targetSize.w/2
        canvas.height = targetSize.h/2
        canvas
          .getContext('2d')
          .drawImage(imageElement, 0, 0, canvas.width, canvas.height)

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
window.runModel = async function () {
  if (imageElement) {
    message('running inference...')
    let img = preprocessInput(imageElement)
    // let start = (new Date()).getTime()
    // https://js.tensorflow.org/api/latest/#tf.Model.predict
    const output = ImageCrop_model.predict(img)
    // let end = (new Date()).getTime()
    segmentData= await processOutput(output)
    // message(`inference ran in ${(end - start) / 1000} secs`, true)
    message(`finish inference`)
    document.getElementById('berrysegment').disabled = false
  } else {
    message('no image available', true)
  }

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
    let contoursColor = new cv.Scalar(255, 255, 255);
    let rectangleColor = new cv.Scalar(255, 0, 0);
    cv.drawContours(dst_mask, contours, i, contoursColor, 1, cv.LINE_8, hierarchy, 100);
    let point1 = new cv.Point(rect.x, rect.y);
    let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
    cv.rectangle(dst_mask, point1, point2, rectangleColor, 1, cv.LINE_AA, 0)

    let rect_resize =Object.assign({},rect);
    rect_resize.x=rect.x*0.98/Ratio1;
    rect_resize.y=rect.y*0.98/Ratio2;
    rect_resize.width=rect.width*1.1/Ratio1;
    rect_resize.height=rect.height*1.15/Ratio2;
    Rect.push(rect_resize);
    }
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

  for (let j=0;j<Rect.length;j++){
    dst_orgImg=src_orgImg.roi(Rect[j]);
    individualBerry.push(dst_orgImg);
    // cv.imshow('individualBerry', dst_orgImg);
  }
  // document.getElementById('berrysegment').disabled = false
}



//crop individual berries
// individualBerry=[];
// window.cropImage = function () {
//   let Ratio1 = imageSize / originalImage.width/2
//   let Ratio2 = imageSize / originalImage.height /2

//   // let canvas_mask = document.getElementById('canvassegments')
//   // let ctx_mask = canvas_mask.getContext("2d");
//   // let imgData_mask = ctx_mask.getImageData(0, 0, canvas_mask.width, canvas_mask.height);
//   // let src_mask = cv.matFromImageData(imgData_mask);
//   let src_mask = cv.matFromImageData(segmentData);
//   let dst_mask= new cv.Mat.zeros(src_mask.rows,src_mask.cols, cv.CV_8UC3);

//   cv.cvtColor(src_mask, src_mask, cv.COLOR_RGBA2GRAY, 0);
//   cv.threshold(src_mask, src_mask, 20, 100, cv.THRESH_BINARY);
//   let contours = new cv.MatVector();
//   let hierarchy = new cv.Mat();
//   cv.findContours(src_mask, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
//   Rect=[];
//   for (let i = 0; i < contours.size(); i++) {
//     let cnt = contours.get(i);
//     let rect = cv.boundingRect(cnt);
//     let contoursColor = new cv.Scalar(255, 255, 255);
//     let rectangleColor = new cv.Scalar(255, 0, 0);
//     cv.drawContours(dst_mask, contours, i, contoursColor, 1, cv.LINE_8, hierarchy, 100);
//     let point1 = new cv.Point(rect.x, rect.y);
//     let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
//     cv.rectangle(dst_mask, point1, point2, rectangleColor, 1, cv.LINE_AA, 0)

//     let rect_resize =Object.assign({},rect);
//     rect_resize.x=rect.x*0.98/Ratio1;
//     rect_resize.y=rect.y*0.98/Ratio2;
//     rect_resize.width=rect.width*1.1/Ratio1;
//     rect_resize.height=rect.height*1.15/Ratio2;
//     Rect.push(rect_resize);
//     }
//   cv.imshow('canvascrop', dst_mask);
//   src_mask.delete(); dst_mask.delete(); contours.delete(); hierarchy.delete();

//   // extract orignial size berry
//   var canvas_orgImg=document.createElement('canvas');
//   var ctx_orgImg = canvas_orgImg.getContext('2d');
//   canvas_orgImg.width = originalImage.width;
//   canvas_orgImg.height = originalImage.height;
//   ctx_orgImg.drawImage(originalImage, 0, 0 );
//   let originalImageData = ctx_orgImg.getImageData(0, 0, canvas_orgImg.width, canvas_orgImg.height);
//   let src_orgImg = cv.matFromImageData(originalImageData);
//   let dst_orgImg= new cv.Mat();

//   for (let j=0;j<Rect.length;j++){
//     dst_orgImg=src_orgImg.roi(Rect[j]);
//     individualBerry.push(dst_orgImg);
//     // cv.imshow('individualBerry', dst_orgImg);
//   }
//   document.getElementById('getResult').disabled = false
// }

function CreatImageData(ImageMat,rx,ry){
  var canvas=document.createElement('canvas');
  let dst = new cv.Mat();
  let dsize = new cv.Size(rx, ry);
  cv.resize(ImageMat, dst, dsize, 0, 0, cv.INTER_AREA);
  cv.imshow(canvas,dst);
  img=canvas.getContext('2d').getImageData(0,0,rx,ry)
  return img;
}

function drawImage(imageData,x,y){
  var canvas=document.getElementById('result1')
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



//berry segmentaion for individual berry

// window.segmentBerry = function () {
var bruise_ratio=[]
window.segmentBerry = async function () {
  let bruiseResult_canvas=document.getElementById('bruiseResult');
  let bruiseResult_ctx=bruiseResult_canvas.getContext('2d');
  bruiseResult_ctx.font = "20px Times New Roman";

  for (var i=0;i<individualBerry.length;i=i+1){
    // for (var i=1;i<2;i=i+1){
    let img=CreatImageData(individualBerry[i],224,224);
    let inputImage = preprocessInput(img);
    const output_berry = berrySeg_model.predict(inputImage);
    count_berry=CountPixel(output_berry);
    // console.log('count_berry:'+count_berry)
    const output_bruise = bruiseSeg_model.predict(inputImage)
    count_bruise=CountPixel(output_bruise);
    // console.log('count_bruise:'+count_bruise)
    if(count_bruise==0){
      ratio=0;
    } else{
      ratio=count_bruise/count_berry;
    };
    bruise_ratio.push(ratio)
    // console.log('i='+i+'  ratio:'+ratio)
    bruiseResult_ctx.fillText('id='+(i)+'  bruiseRatio='+ratio.toFixed(2) , 0, i*25);

    // var width=individualBerry[i].cols;
    // var height= individualBerry[i].rows;


    // let Ori_berry=CreatImageData(individualBerry[i],width,height);
    // drawImage(Ori_berry,0,0)

    // let berry_imageData= await processOutput(output_berry)
    // let mat_berry = cv.matFromImageData(berry_imageData);
    // let Ori_berry_mask=CreatImageData(mat_berry,width,height);
    // drawImage(Ori_berry_mask,width+10,0)
    // // // CreatImage(Ori_berry_mask)

    // let brusie_imageData= await processOutput(output_bruise)
    // let mat_bruise = cv.matFromImageData(brusie_imageData);
    // let Ori_bruise_mask=CreatImageData(mat_bruise,width,height);
    // drawImage(Ori_bruise_mask,2*width+2*10,0)


    // CreatImage(Ori_bruise_mask)
  }


  
  


}



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
async function processOutput (output) {
  // console.log('processOutput started')

  let segMap = Array.from(output.dataSync())
  console.log('segMap:'+segMap)

  if (!colorMap) {
    await loadColorMap()
  }

  let segMapColor = segMap.map(seg => colorMap[(seg>0.8)?1:0])

  // let canvas = document.getElementById('canvassegments')
  let canvas = document.createElement('canvas')
  let ctx = canvas.getContext('2d')
  canvas.width = targetSize.w/2
  canvas.height = targetSize.h/2
  console.log('segMaColor:'+segMapColor)
  let data = []
  for (var i = 1; i < segMapColor.length; i=i+2) {
    data.push(segMapColor[i][0]) // red
    data.push(segMapColor[i][1]) // green
    data.push(segMapColor[i][2]) // blue
    data.push(175) // alpha
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
  if (mark) {
    node.appendChild(mark)
  } else {
    node.innerText = msg
  }

  document.getElementById('message').appendChild(node)
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

