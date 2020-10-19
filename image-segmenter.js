/* global tf, Image, FileReader, ImageData, fetch */

const ImageCropUrl = 'model/ImageCrop/model.json'
const berrySegUrl = 'model/berrySegmentation/model.json'
const bruiseSegUrl = 'model/bruiseSegmentation/model.json'
const colorMapUrl = 'assets/color-map.json'

//const imageSize = 512
const imageSize_0 = 300
const imageSize = 224 // for segmentation

let targetSize_0 = { w: imageSize_0, h: imageSize_0}
let targetSize = { w: imageSize, h: imageSize }
let model
let imageElement
let colorMap
let output

// threshold
const thr_1=0.5
const thr_2=0.2
const thr_3=0.2



/**
 * load the TensorFlow.js model
 */
window.loadModel = async function () {
  document.getElementById('message1').innerHTML = "";
  message('message1','loading model...')

  let start1 = (new Date()).getTime()
  ImageCrop_model = await tf.loadGraphModel(ImageCropUrl)
  let end1 = (new Date()).getTime()
  message('message1',`ImageCrop_model loaded in ${(end1 - start1) / 1000} secs`, true)

  let start2 = (new Date()).getTime()
  berrySeg_model = await tf.loadGraphModel(berrySegUrl)
  // berrySeg_model.getWeights()[0].print()
  let end2 = (new Date()).getTime()
  message('message1',`berrySeg_model loaded in ${(end2 - start2) / 1000} secs`, true)

  let start3 = (new Date()).getTime()
  // bruiseSeg_model = await tf.loadGraphModel(bruiseSegUrl)
  bruiseSeg_model = await tf.loadLayersModel(bruiseSegUrl)
  // bruiseSeg_model.getWeights()[0].print()

  let end3 = (new Date()).getTime()
  message('message1',`bruiseSeg_model loaded in ${(end3 - start3) / 1000} secs`, true)
  document.getElementById('modelimage').disabled = false
  // enableElements()
}


originalImage=new Image()
/**
 * handle image upload
 *
 * @param {DOM Node} input - the image file upload element
 */
window.loadImage = function (input) {
  individualBerry=[];
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
  document.getElementById('message2').innerHTML = "";
  document.getElementById('message3').innerHTML = "";
  document.getElementById('message4').innerHTML = "";
  document.getElementById('message5').innerHTML = "";
  document.getElementById('message6').innerHTML = "";
  document.getElementById('message7').innerHTML = "";
  document.getElementById('message8').innerHTML = "";

  if (input.files && input.files[0]) {
    // disableElements('input:'+input.files )
    message('message2','resizing image...')

    let reader = new FileReader()

    reader.onload = function (e) {
      let src = e.target.result

      document.getElementById('canvasimage').getContext('2d').clearRect(0, 0, targetSize_0.w, targetSize_0.h)
      imageElement = new Image()
      imageElement.src = src
      originalImage=imageElement.cloneNode();

      imageElement.onload = function () {
        let resizeRatio1 = imageSize_0 / imageElement.width
        let resizeRatio2 = imageSize_0 / imageElement.height
        targetSize_0.w = Math.round(resizeRatio1 * imageElement.width)
        targetSize_0.h= Math.round(resizeRatio2 * imageElement.height)

        let origSize = {
          w: imageElement.width,
          h: imageElement.height
        }
        imageElement.width = targetSize_0.w
        imageElement.height = targetSize_0.h

        let canvas = document.getElementById('canvasimage')
        let ctx=canvas.getContext('2d')
        ctx.drawImage(imageElement, 0, 0,targetSize_0.w,targetSize_0.h)
        message('message2',`resized from ${origSize.w} x ${origSize.h} to ${targetSize_0.w} x ${targetSize_0.h}`)
      }
    }

    reader.readAsDataURL(input.files[0])
  } else {
    message('message2','no image uploaded', true)
  }
  document.getElementById('modelrun').disabled = false
}



/**
 * run the model and get a prediction
 */
individualBerry=[];
var berry_resize_ratio
window.runModel = async function () {
  message('message2','running cropping...')
  let start = (new Date()).getTime()
  Rect=[];
  if (imageElement) {
    // https://js.tensorflow.org/api/latest/#tf.Model.predict
    let output = await ImageCrop_model.executeAsync(tf.browser.fromPixels(originalImage).resizeNearestNeighbor([300, 300]).expandDims());
    let score= output[1].arraySync()
    console.log('score',score)
    let label = output[2].arraySync()
    let bbx = output[0].arraySync() //[ymin xmin ymax xmax]
    var c = document.getElementById("canvasimage");
    var ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.lineWidth = "0.5";
    ctx.strokeStyle = "blue";
    for (let i = 0; i < score[0].length; i++) {
      if (score[0][i]>thr_1) {
        // draw boxes
        x0=bbx[0][i][1]*300//-1
        y0=bbx[0][i][0]*300//-1
        bb_width=(bbx[0][i][3]-bbx[0][i][1])*300//+2
        bb_height=(bbx[0][i][2]-bbx[0][i][0])*300//+2
        ctx.rect(x0,y0,bb_width,bb_height);        
        ctx.stroke();
        //store individual berries
        let temp=new Object();
        temp.x=bbx[0][i][1]*originalImage.width//-originalImage.width/300*2
        temp.y=bbx[0][i][0]*originalImage.height//-originalImage.height/300*2
        temp.width=(bbx[0][i][3]-bbx[0][i][1])*originalImage.width//+originalImage.width/300*4
        temp.height=(bbx[0][i][2]-bbx[0][i][0])*originalImage.height//+originalImage.height/300*4
        Rect.push(temp)
      }
    }
  }
  
  // sort berries in order
  Rect.sort((a, b) => (a.y > b.y) ? 1 : -1 )
  var meanHeight = Rect.reduce(function(prev, cur) {return prev + cur.height; }, 0)/Rect.length;
  k_temp=[];
  k_temp.push(0)
  for (k=1;k<Rect.length;k++){
    if((Rect[k].y-Rect[k-1].y)>meanHeight/2){
      k_temp.push(k);
    }    
  }
  Rect_1=[]
  RRect=[]
  for (m=0;m<k_temp.length;m++){
    Rect_1=Rect.slice(k_temp[m],k_temp[m+1])
    Rect_1.sort((a, b) => (a.x > b.x) ? 1 : -1 )
    for (n=0;n<Rect_1.length;n++){
      RRect.push(Rect_1[n])
    }
  }
  for (let j=0;j<RRect.length;j++){
    x_0=RRect[j].x/originalImage.width*300
    y_0=RRect[j].y/originalImage.height*300
    ctx.fillText(j+1,x_0,y_0)
    individualBerry.push(RRect[j]);

  }

  let end = (new Date()).getTime()
  message('message2',`finish cropping in ${(end - start) / 1000} secs`, true)
  document.getElementById('berrysegment').disabled = false
}


//berry segmentaion for individual berry
var bruise_ratio=[]
window.segmentBerry = async function () {
  message('message2',`start segmentation...`)
  let start = (new Date()).getTime()

  let ratio_canvas=document.getElementById('bruiseResult');
  let ratio_ctx=ratio_canvas.getContext('2d');
  ratio_ctx.font = "15px Times New Roman";
  ratio_ctx.fillText('' ,0, 0);

  let mask_canvas=document.getElementById('origin');
  let mask_ctx=mask_canvas.getContext('2d');
  mask_ctx.drawImage(originalImage, 0, 0,mask_canvas.width,mask_canvas.height)
  mask_ctx.beginPath();
  mask_ctx.lineWidth = "0.5";
  mask_ctx.strokeStyle = "blue";
  
  // predict result
  for (var i=0;i<individualBerry.length;i++){
    var canvas=document.createElement('canvas');
    var ctx=canvas.getContext('2d')
    canvas.width=originalImage.width;
    canvas.height=originalImage.height;
    ctx.drawImage(originalImage,0,0,canvas.width,canvas.height)
    let img=ctx.getImageData(individualBerry[i].x,individualBerry[i].y,individualBerry[i].width,individualBerry[i].height);
    let inputImage = tf.browser.fromPixels(img).toFloat().resizeNearestNeighbor([224, 224]).expandDims();
    const output_berry = berrySeg_model.predict(inputImage);
    count_berry=CountPixel(output_berry,thr_2);
    const output_bruise = bruiseSeg_model.predict(inputImage)
    count_bruise=CountPixel(output_bruise,thr_3);
    if(count_bruise==0){
      ratio=0;
    } else{
      ratio=count_bruise/count_berry;
    };
    if (i<10){
      if (ratio<1 | ratio==1){
        message('message3','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message3','id='+(i+1)+':100%')
      }     
    } else if(i<20){
      if (ratio<1 | ratio==1){
        message('message4','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message4','id='+(i+1)+':100%')
      }  
    } else if(i<30){
      if (ratio<1 | ratio==1){
        message('message5','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message5','id='+(i+1)+':100%')
      }  
    } else if(i<40){
      if (ratio<1 | ratio==1){
        message('message6','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message6','id='+(i+1)+':100%')
      }  
    } else if(i<50){
      if (ratio<1 | ratio==1){
        message('message7','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message7','id='+(i+1)+':100%')
      }  
    } else if(i<60){
      if (ratio<1 | ratio==1){
        message('message8','id='+(i+1)+': '+Math.floor(ratio*100)+'%')
      } else {
        message('message8','id='+(i+1)+':100%')
      }  
    }
    

    bruise_ratio.push(Math.floor(ratio*100))
    let berry_imageData= await processOutput(output_berry,[0,255,0,100],thr_2)
    let bruise_imageData= await processOutput(output_bruise,[255,0,0,255],thr_3)
    let mat_berry = cv.matFromImageData(berry_imageData);
    let mat_bruise = cv.matFromImageData(bruise_imageData);
    resize_ratio_1=Math.round(individualBerry[i].width*300/originalImage.width)
    resize_ratio_2=Math.round(individualBerry[i].height*300/originalImage.height)
    let berry_mask=CreatImageData(mat_berry,resize_ratio_1,resize_ratio_2);
    let bruise_mask=CreatImageData(mat_bruise,resize_ratio_1,resize_ratio_2);
    x0=individualBerry[i].x*300/originalImage.width;
    y0=individualBerry[i].y*300/originalImage.height;

    mask_ctx.rect(x0,y0,resize_ratio_1,resize_ratio_2);
    mask_ctx.stroke();
    drawImage('berrySegmentation',berry_mask,x0,y0)
    drawImage('bruiseSegmentation',bruise_mask,x0,y0)
  }
  let end = (new Date()).getTime()
  message('message2',`finish segmentation in ${(end - start) / 1000} secs`, true)
  document.getElementById('download').disabled = false
}

function downloadFile() {
  var obj = bruise_ratio;
  var filename = "download.json";
  var blob = new Blob([JSON.stringify(obj)], {type: 'text/plain'});
  if (window.navigator && window.navigator.msSaveOrOpenBlob) {
      window.navigator.msSaveOrOpenBlob(blob, filename);
  } else{
      var e = document.createEvent('MouseEvents'),
      a = document.createElement('a');
      a.download = filename;
      a.href = window.URL.createObjectURL(blob);
      a.dataset.downloadurl = ['text/plain', a.download, a.href].join(':');
      e.initEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
      a.dispatchEvent(e);
  }
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


/**
 * convert image to Tensor input required by the model
 *
 * @param {HTMLImageElement} imageInput - the image element
 */
function preprocessInput (imageInput) {
  let inputTensor = tf.browser.fromPixels(imageInput).toFloat()
  // https://js.tensorflow.org/api/latest/#expandDims
  let preprocessed = inputTensor.expandDims()
  return preprocessed
}

function CountPixel (output,threshold) {
  let segMap = Array.from(output.dataSync())
  var count=0;
  for (j=1;j<segMap.length;j=j+2){
    if(segMap[j]>threshold){
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
async function processOutput (output,color,threshold) {
  let segMap = Array.from(output.dataSync())
  // console.log('segMap',segMap)
  segMapColor=[]
  for(var i=0;i<segMap.length;i++){
    if (segMap[i]>threshold){
      segMapColor.push(color)
    } else {
      segMapColor.push([0,0,0,0])
    }
  }
  let canvas = document.createElement('canvas')
  let ctx = canvas.getContext('2d')
  canvas.width = targetSize.w/2
  canvas.height = targetSize.h/2
  let data = []
  for (var i = 1; i < segMapColor.length; i=i+2) {
    data.push(segMapColor[i][0]) // red
    data.push(segMapColor[i][1]) // green
    data.push(segMapColor[i][2]) // blue
    data.push(segMapColor[i][3]) // alpha
  }
  let imageData = new ImageData(canvas.width, canvas.height)
  imageData.data.set(data)
  ctx.putImageData(imageData, 0, 0)

  return imageData
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

function message (id,msg, highlight) {
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
  
  document.getElementById(id).appendChild(node)
  // document.getElementById('message')

}

function init () {
  message('message1', `tfjs version: ${tf.version.tfjs}`, true)
}

// ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  setTimeout(init, 500)
}

