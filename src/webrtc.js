export function getWebRTCVideoAsync(){
  return new Promise(function (res, rej){
    try {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }).then(function (stream){
        let video = document.createElement('video');
        video.srcObject = stream;
        video.setAttribute('playsinline', '');
        video.onloadedmetadata = function (){
          video.play();
          res(video);
        };
        video.addEventListener('error', rej);
      })
    } catch (ex) {
      rej(ex);
    }
  })
}