import { FeatTrainer } from './FeatTrainer';
import { loadImageAsync, $ } from './common';
import { getWebRTCVideoAsync } from './webrtc';

let trainer = new FeatTrainer();
let pattern;
let cvs = document.createElement('canvas');
let ctx = cvs.getContext('2d');

loadImageAsync($('.img')).then(img => {
  let grayImg = trainer.getGrayScaleMat(img);
  pattern = trainer.trainPattern(grayImg);
  return getWebRTCVideoAsync();
}).then(function (video) {
  cvs.width = video.videoWidth;
  cvs.height = video.videoHeight;
  cvs.style.width = document.documentElement.clientWidth + 'px';
  cvs.style.height = document.documentElement.clientHeight + 'px';
  document.body.appendChild(cvs);
  let mVideo = $('.video');

  function loop() {
    drawVideo(video, mVideo);
    requestAnimationFrame(loop)
  }

  loop();
  mVideo.play();
});

window.addEventListener('touchstart', function () {
  let video = $('.video');
  video.play();
});
let lastTransform;

function drawVideo(video, mVideo) {
  if (video.readyState === 4) {
    ctx.drawImage(video, 0, 0);
    if (pattern) {
      let grayImage = trainer.getGrayScaleMat(ctx.getImageData(0, 0, video.videoWidth, video.videoHeight));
      let features = trainer.describeFeatures(grayImage);
      let matches = trainer.matchPattern(features.descriptors, pattern.descriptors);
      const result = trainer.findTransform(matches, features.keyPoints, pattern.keyPoints);
      let transform = result && result.goodMatch > 8 ? result.transform : lastTransform;

      if (transform) {
        ctx.save();
        let [ a, c, e, b, d, f ] = transform.data;
        let size = pattern.levels[ 0 ];

        ctx.transform(a, b, c, d, e, f);
        ctx.drawImage(mVideo, 0, 0, mVideo.videoWidth, mVideo.videoHeight, 0, 0, size[ 0 ], size[ 1 ]);
        ctx.restore();
        lastTransform = transform;
      }
    }
  }
}


