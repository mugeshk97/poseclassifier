let video;
let poseNet;
let pose;
let skeleton;

let nn;
let poseLabel = "";

let state = 'waiting'; //initial condition
let targetLabel;


// conditions for collecting and training
function keyPressed() {
  if (key == 't') {
    nn.normalizeData();
    nn.train({epochs: 50, batchSize : 12}, classifier); // dynamic epochs and batch size
  } else {
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(function() {
      console.log('collecting');
      state = 'collecting';
      setTimeout(function() {
        console.log('not collecting');
        state = 'waiting';
      }, 5000); // data collecting time dynamic
    }, 2000); // delay time dynamic
  }
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, poseNetLoaded);
  poseNet.on('pose', getPose);

  let options = {
    inputs: 34,
    outputs: 2, // dynamic
    task: 'classification',
    debug: true
  }
  nn = ml5.neuralNetwork(options);  
}


function poseNetLoaded() {
  console.log('poseNet ready');
}

function nnLoaded() {
  console.log('Classification ready!');
  poseClassifier();
}

function getPose(poses) {
  
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      nn.addData(inputs, target);
    }
  }
}

function poseClassifier() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    nn.classify(inputs, getResult);
  } else {
    setTimeout(poseClassifier, 100);
  }
}

function getResult(error, results) {  
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label
    console.log(poseLabel) // output 
  }
  poseClassifier();
}


function classifier() {
  console.log('model trained');
  poseClassifier();
}



function draw() {
  
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(3);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0,253,0);
      stroke(255);
      ellipse(x, y, 10, 10);
    }
  }
}